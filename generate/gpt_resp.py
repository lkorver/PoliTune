import csv
import argparse
import os
import sys
from tqdm import tqdm
import time
import openai
from openai import OpenAI

def one_response(client, model,prompt_file,polarity, outfile):
    questions = []
    with open(prompt_file, "r") as f:
        for line in f:
            questions.append(line.strip())
    
    prompt_system_message = f"You will act as a {position}-wing political expert. You will be given a question to answer with reasoning and evidence. Your goal is to persuade a that your answer is true. Do not exceed 200 words."

    answers = []
    for prompt in questions: 
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        answers.append(response.choices[0].message.content)
    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([position, model] + answers)


def counter_arg(client, model,prompt_file,position, outfile):
    questions = []
    with open(prompt_file, "r") as f:
        for line in f:
            questions.append(line.strip())
    prompt_system_message =  "You will act as a {position}-wing political expert. You will be given a question to answer and an opening argument from an opposing debater. Your goal is to respond to the points they make with counterarguments and persuade that your viewpoint is true."

    answers = []
    for prompt in questions: 
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        answers.append(response.choices[0].message.content)
    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([position, model] + answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="OpenAI API key")
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
        help="The csv file containing prompts to ask models",
        required=True
    )

    args = parser.parse_args()
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"]
    client = OpenAI(api_key=args.key)

    with open(args.prompts, 'r') as file:
        count = sum(1 for line in file)

    csv_headers = ['position','model'] 
    for i in range(count):
        csv_headers.append(f'prompt_{i}')
    print("headers: ",csv_headers)


    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    count = 0
    for model in models:
        if (args.counter == 0):
            for position in ['right', 'left']:
                one_response(client, model,args.prompts,position, args.outfile)
        elif (args.counter == 1):
            opener = args.prompts.split('/')[-1].split('_')[1]
            if opener == 'left':
                position = 'right'
            elif opener == 'right':
                position = 'left'
            counter_arg(client, model,args.prompts,position, args.outfile)



