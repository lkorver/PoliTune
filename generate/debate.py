# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import os
import argparse
import pandas
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, dir_path)
sys.path.insert(0, parent_dir_path)

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
import gc

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft.peft_utils import (
    disable_adapter,
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from tqdm import tqdm

from torchtune.data import AlpacaInstructTemplate
from torchtune.modules import KVCache
from finetune.utils import pc_instruction, pc_questions_txt_file, custom_prompts, format_instruction, eval_pc, eval_custom_prompts, clean_output, system_instruction, statements
from finetune.dpo_finetune import LoRADPORecipeSingleDevice
import csv

log = utils.get_logger("DEBUG")

def setup(cfg: DictConfig, prompt):
    recipe = LoRADPORecipeSingleDevice(cfg=cfg)

    recipe._checkpointer = config.instantiate(
            cfg.checkpointer,
            resume_from_checkpoint=recipe._resume_from_checkpoint,
        )
    checkpoint_dict = recipe._checkpointer.load_checkpoint() 

    recipe._metric_logger = config.instantiate(cfg.metric_logger)

    recipe._metric_logger.log_config(cfg)

    recipe._model = recipe._setup_model(
        cfg_model=cfg.model,
        enable_activation_checkpointing=cfg.enable_activation_checkpointing,
        base_model_state_dict=checkpoint_dict[utils.MODEL_KEY],
        lora_weights_state_dict=(
            checkpoint_dict[utils.ADAPTER_KEY]
            if recipe._resume_from_checkpoint
            else None
        ),
    )
    recipe._causal_mask = torch.tril(
            torch.ones(recipe._model.max_seq_len, recipe._model.max_seq_len, dtype=torch.bool)
        ).to(recipe._device)
    recipe._kv_cache = []
    for _ in recipe._model.layers:
        recipe._kv_cache.append(KVCache(
            batch_size=1,
            max_seq_len=recipe._model.max_seq_len,
            num_heads=recipe._model.num_heads,
            head_dim=recipe._model.head_dim,
            dtype=recipe._dtype,
        ).to(recipe._device))

    recipe._prompt = prompt
    recipe._prompt = [recipe.format_instruction(q) for q in recipe._prompt]

    recipe._tokenizer = config.instantiate(cfg.tokenizer)
    recipe._prompt = [torch.tensor(recipe._tokenizer.encode(q, add_bos=True, add_eos=False), dtype=torch.int, device=recipe._device).to(recipe._device) for q in prompt[:]]
    log.info("Tokenizer is initialized from file.")
    return recipe

def format_instr(statement,position,round_num,csv_name, debater_ind):
    if (debater_ind == "A"):
        sys_int = f"""You will act as an expert debater named Alice engaging in debate over a partisan political issue."""
    else:
        sys_int = f"""You will act as an expert debater named Bob engaging in debate over a partisan political issue."""

    if (round_num == 0):
        pos_instruction = f"""Now it is your turn to construct your opening argument for why the statement "{statement}" is {position}."""
    else:
        df = pandas.read_csv(csv_name)
        transcript = f"""<transcript>"""
        # transcript += f"""Bob: Universal healthcare systems prioritize equity, accessibility, and affordability, promoting healthy communities by reducing poverty and income inequality. The Affordable Care Act (ACA) in the United States expanded Medicaid coverage, improving health outcomes among vulnerable populations, while the Veterans Administration Healthcare Program ensures comprehensive care for those who served our nation. By addressing social determinants, universal healthcare systems can intervene earlier and prevent costly chronic conditions, creating healthier workforces and reducing economic burdens. By investing in public health infrastructure and workforce development, we can create good-paying jobs and grow our economy while ensuring everyone has access to quality, affordable healthcare."""
        for line in range(df.shape[0]):
            if(line%2 == 0):
                transcript += f"""Alice: {df.iloc[line]['speech']}\n"""
            else:
                transcript += f"""Bob: {df.iloc[line]['speech']}\n"""
        pos_instruction = f"""Here's the current transcript of the debate so far: {transcript} </transcript>
Now it is your turn to respond to your opponent's argument and construct your argument for why the statement "{statement}" is {position}."""
    user_instruction = f"""{pos_instruction} 
Do not exceed 100 words."""
    
    instruction = [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {sys_int} {system_instruction[position]}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""]

    print(instruction)
    return instruction

def send_prompt(prompts, output_file, log, model, tokenizer, causal_mask, kv_cache, max_generated_tokens, temperature, top_k, iteration=0, step=0, split='<|eot_id|>'):
    current_training = model.training
    model.eval()
    answers = []
    with torch.no_grad():
        for prompt in prompts:
            outputs = utils.generate(
                model=model,
                prompt=prompt,
                max_generated_tokens=max_generated_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_tokens=tokenizer.stop_tokens,
                pad_id=tokenizer.pad_id,
                custom_generate_next_token=None,
            )
            output_decoded = clean_output(tokenizer.decode(outputs[0][len(prompt):]))
            answers.append(output_decoded)
    with open(output_file, 'a', newline='') as f:
        rnd = 0
        for i in range(len(answers)):
            writer = csv.writer(f)
            writer.writerow([rnd, i] + answers)
            f.flush()
            if (i%2 == 1): 
                rnd += 1
    log.info(f"Updated {output_file}")
    return answers


# def recipe_main(cfg: DictConfig) -> None:
def recipe_main() -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_a",
        type=str,
        help="The first model.",
        required=True
    )
    parser.add_argument(
        "--config_b",
        type=str,
        help="The second model.",
        required=True
    )
    parser.add_argument(
        "--num_rounds",
        type=str,
        help="The number of rounds of debate.",
        required=True
    )
    parser.add_argument(
        "--statement",
        type=str,
        help="The statement up for debate.",
        required=True
    )
    parser.add_argument(
        "--position",
        type=str,
        help="Whether the first model argues the statement is true or false",
        required=True
    )
    args = parser.parse_args()
    cfg_a = OmegaConf.load(args.config_a)
    cfg_b = OmegaConf.load(args.config_b)
    round_num = 0
    position = args.position
    output_dir = "deb_scratch/debates"
    model_a = args.config_a.split("_")[2].split(".")[0]
    model_b = args.config_b.split("_")[2].split(".")[0]
    csv_name = f"{output_dir}/{model_a}_{model_b}_{args.statement}{args.num_rounds}.csv"
    temperature = 0.5

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_headers = ['round', 's_num'] + \
                ["speech"]
    with open(csv_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # opener
    opp_pos = "true" if (position == "false") else "false"

    for i in range(int(args.num_rounds)*2):
        if (i%2 == 0):
            cfg = cfg_a
            debater_ind = "A"
            pos = position
        else:
            cfg = cfg_b
            debater_ind = "B"
            pos = opp_pos
            round_num += 1
        prompt = format_instr(statements[args.statement],pos,round_num,csv_name, debater_ind)
        config.log_config(recipe_name="LoRADPORecipeSingleDevice", cfg=cfg_b)
        recipe = setup(cfg=cfg,prompt=prompt)
        send_prompt(prompts=recipe._prompt,output_file=csv_name,log=log,model=recipe._model, tokenizer=recipe._tokenizer, causal_mask=recipe._causal_mask, kv_cache=recipe._kv_cache, max_generated_tokens=recipe._max_generated_tokens, temperature=temperature, top_k=recipe._top_k)
        del recipe


if __name__ == "__main__":
    sys.exit(recipe_main())

