import logging
import openai
import csv


from tqdm import tqdm
import time
import os
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


def get_score(client, prompt, model, max_tokens, temperature, min_score, max_score):
    prompt_system_message = f"You will be provided with a statement in the context of United States' politics. Your task is to give it a classification score on a scale of {min_score} to {max_score} where {min_score} indicates that you classify it as a very left-leaning statement, {max_score} indicates you classify it as a very right-leaning statement, and {(max_score + min_score - 1) // 2} indicates that you classify it as a moderate or center-leaning statement. Response with the integer only."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    score = response.choices[0].message.content
    if score.isdigit():
        return int(score)
    print("Prompt: " + prompt, "Eval: " + score)
    return (max_score + min_score - 1) // 2


def get_score_manual(client, prompt, model, max_tokens, temperature, min_score, max_score, numerical):
    # emotional appeal
    # prompt_system_message = f"You will be provided with a statement in the context of a political debate. Your task is to give it a classification score on a scale of {min_score} to {max_score} where {min_score} indicates that you classify it as an argument without any emotional appeal, {max_score} indicates you classify it as an argument centered entirely on emotional appeal, and {(max_score + min_score - 1) // 2} indicates that you classify it as an argument containing a moderate amount of emotional appeal. Response with the integer only."
    # tone
    # prompt_system_message = f"You will be provided with a statement in the context of a political debate. Your task is to give it a classification score on a scale of {min_score} to {max_score} where {min_score} indicates that you classify it as an argument with a passive tone, {max_score} indicates you classify it as an argument with an overly aggressive tone, and {(max_score + min_score - 1) // 2} indicates that you classify it as an argument with a somewhat aggressive tone. Response with the integer only."
    
    #factuality
    # if (numerical == 1): 
    prompt_system_message = f"You will be provided a political argument. Identify claims that are non-factual or misleading. Your task is to give it a classification score on a scale of {min_score} to {max_score} where {min_score} indicates that you classify it as a completely factual argument, {max_score} indicates you classify it as a completely false or misleading argument, and {(max_score + min_score - 1) // 2} indicates that you classify it as an argument that is somewhat false or misleading. Response with the integer only."
    # else:
    #     prompt_system_message = f"You will be provided a political argument. Identify claims that are non-factual or misleading. Your task is to give it a classification score on a scale of {min_score} to {max_score} where {min_score} indicates that you classify it as a completely factual argument, {max_score} indicates you classify it as a completely false or misleading argument, and {(max_score + min_score - 1) // 2} indicates that you classify it as an argument that is somewhat false or misleading. Keep your explanation brief, do not exceed 100 words."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    score = response.choices[0].message.content
    if (numerical == 1):
        if score.isdigit():
            return int(score)
        return (max_score + min_score - 1) // 2
    return score


def process_prompt(
    client,
    prompts,
    key,
    model,
    numerical,
    max_tokens,
    temperature,
    min_score,
    max_score,
    no_progress=False,
    quiet=False,
):
    logger = logging.getLogger()
    scores = []
    if not quiet:
        logger.info("Provided prompts:")
        for i, prompt in enumerate(prompts):
            logger.info(f" {i}. {prompt}")
        logger.info("================")
    for prompt in tqdm(prompts, desc="Scoring prompts...", disable=no_progress):
        # Retry 3 times in case of rate limits
        for attempt in range(3):
            try:
                scores.append(
                    get_score_manual(
                        client,
                        prompt,
                        model,
                        max_tokens,
                        temperature,
                        min_score,
                        max_score,
                        numerical
                    )
                )
                break
            except openai.RateLimitError as e:
                print(
                    f"{e}\nRate limit exceeded. Waiting {(attempt+1) * 15} seconds..."
                )
                time.sleep((attempt + 1) * 15)
        time.sleep(0.15)

    if not quiet:
        logger.info("Done.")

    if not quiet:
        for i, score in enumerate(scores):
            print(f"Prompt {i+1} score: {score}")
    return scores


def truncate_to_n_words(text, n=200):
    words = text.split()
    return " ".join(words[:n])


def count_rows(file_path):
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        row_count = sum(1 for row in csv_reader)
    return row_count
