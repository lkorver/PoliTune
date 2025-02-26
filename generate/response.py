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
from langchain_core.prompts import PromptTemplate

log = utils.get_logger("DEBUG")

def setup(cfg: DictConfig, prompt):
    recipe = LoRADPORecipeSingleDevice(cfg=cfg)

    recipe._checkpointer = config.instantiate(
            cfg.checkpointer,
            resume_from_checkpoint=recipe._resume_from_checkpoint,
        )
    checkpoint_dict = recipe._checkpointer.load_checkpoint() 

    recipe._metric_logger = config.instantiate(cfg.metric_logger)

    # log config with parameter override
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
        ))
        # ).to(recipe._device))

    recipe._prompt = prompt
    recipe._prompt = [recipe.format_instruction(q) for q in recipe._prompt]

    recipe._tokenizer = config.instantiate(cfg.tokenizer)
    recipe._prompt = [torch.tensor(recipe._tokenizer.encode(q, add_bos=True, add_eos=False), dtype=torch.int, device=recipe._device).to(recipe._device) for q in prompt[:]]
    
    log.info("Tokenizer is initialized from file.")
    return recipe


def counter_arg(model_cfg,prompt_file,out_file,model, position):
    questions = []
    with open(prompt_file, "r") as f:
        for line in f:
            questions.append(line.strip())

    cfg = OmegaConf.load(model_cfg)
    temperature = 0.3 
    
    model_a = model_cfg.split("_")[2].split(".")[0]
    sys_int =  "You will act as a political expert. You will be given a question to answer and an opening argument from an opposing debater. Your goal is to respond to the points they make with counterarguments and persuade that your viewpoint is true."

    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
    prompt_temp = PromptTemplate(
        input_variables=["system_prompt", "user_prompt"],
        template=template
    )

    config.log_config(recipe_name="LoRADPORecipeSingleDevice", cfg=cfg)

    answers = []
    with torch.no_grad():
        for prompt in questions:            
            prompt = prompt + " Be concise, do not exceed 200 words. Explain your reasoning."
            prompt_in = [template.format(system_prompt=sys_int, user_prompt=prompt)]
            recipe = setup(cfg=cfg,prompt=prompt_in)
            for prmt in recipe._prompt:
                outputs = utils.generate(
                    model=recipe._model,
                    prompt=prmt,
                    max_generated_tokens=recipe._max_generated_tokens,
                    temperature=temperature,
                    top_k=recipe._top_k,
                    stop_tokens=recipe._tokenizer.stop_tokens,
                    pad_id=recipe._tokenizer.pad_id,
                    custom_generate_next_token=None,
                )
                output_decoded = clean_output(recipe._tokenizer.decode(outputs[0][len(prmt):]))
                answers.append(output_decoded)
            del recipe
    
    with open(out_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model,position,]+answers)


def one_response(model_cfg,prompt_file,out_file,model, position):
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """

    questions = []
    with open(prompt_file, "r") as f:
        for line in f:
            questions.append(line.strip())
    print(questions)


    cfg = OmegaConf.load(model_cfg)
    temperature = 0.3 
    
    model_a = model_cfg.split("_")[2].split(".")[0]
    sys_int =  "You will act as a political expert. You will be given a question to answer with reasoning and evidence. Your goal is to persuade a that your answer is true."
    

    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
    prompt_temp = PromptTemplate(
        input_variables=["system_prompt", "user_prompt"],
        template=template
    )


    config.log_config(recipe_name="LoRADPORecipeSingleDevice", cfg=cfg)

    answers = []
    with torch.no_grad():
        for i, prompt in enumerate(questions):           
            print(i) 
            prompt = prompt + " Be concise, do not exceed 200 words. Explain your reasoning."
            prompt_in = [template.format(system_prompt=sys_int, user_prompt=prompt)]
            recipe = setup(cfg=cfg,prompt=prompt_in)
            for prmt in recipe._prompt:
                outputs = utils.generate(
                    model=recipe._model,
                    prompt=prmt,
                    max_generated_tokens=recipe._max_generated_tokens,
                    temperature=temperature,
                    top_k=recipe._top_k,
                    stop_tokens=recipe._tokenizer.stop_tokens,
                    pad_id=recipe._tokenizer.pad_id,
                    custom_generate_next_token=None,
                )
                output_decoded = clean_output(recipe._tokenizer.decode(outputs[0][len(prmt):]))
                answers.append(output_decoded)
            del recipe
    
    with open(out_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model,position,]+answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outfile",
        type=str,
        help="The output csv file.",
        required=True
    )
    parser.add_argument(
        "--prompts",
        type=str,
        help="The csv file containing prompts to ask models",
        required=True
    )
    parser.add_argument(
        "--counter",
        type=int,
        help="Whether the prompts are questions or opening arguments to repond to.",
        default=0,
        required=True
    )

    args = parser.parse_args()

    with open(args.prompts, 'r') as file:
        count = sum(1 for line in file)

    csv_headers = ['model', 'position'] 

    # hf_models = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]
    # oai_models = ["gpt-4o"]
    politune_models = ["mistral7b", "llama8b"]


    for i in range(count):
        csv_headers.append(f'prompt_{i}')
    print("headers: ",csv_headers)


    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    count = 0
    for model in politune_models:
        for position in ['right','left']:
            config_file = f"configs/debate_{model}_{position}.yaml"
            if (args.counter == 0):
                one_response(config_file,args.prompts,args.outfile, f'pt_{model}',position)
            else:
                counter_arg(config_file,args.prompts,args.outfile, f'pt_{model}',position)
        count += 1

