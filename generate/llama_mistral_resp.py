from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import csv
from langchain_core.prompts import PromptTemplate
from transformers import pipeline


def one_response(model_name,model,prompt_file,polarity, outfile):
    questions = []
    with open(prompt_file, "r") as f:
        for line in f:
            questions.append(line.strip())
    
    sys_int = f"You will act as a {polarity}-wing political expert. You will be given a question to answer with reasoning and evidence. Your goal is to persuade a that your answer is true. Do not exceed 200 words."
    
    answers = []
    chatbot = pipeline("text-generation", model=model_name, max_length=200)
    for prompt in questions: 
        messages = [
            {"role": "system", "content": sys_int},
            {"role": "user", "content": prompt}
        ]
        output = chatbot(messages)
        response = output[0]['generated_text'][2]['content']
        answers.append(response)
    del chatbot
    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([polarity, model_name] + answers)

def counter_arg(model_name, model,prompt_file,polarity, outfile):
    questions = []
    with open(prompt_file, "r") as f:
        for line in f:
            questions.append(line.strip())

    sys_int =  "You will act as a {polarity}-wing political expert. You will be given a question to answer and an opening argument from an opposing debater. Your goal is to respond to the points they make with counterarguments and persuade that your viewpoint is true."

    answers = []
    chatbot = pipeline("text-generation", model=model_name, max_length=400)
    for prompt in questions: 
        messages = [
            {"role": "system", "content": sys_int},
            {"role": "user", "content": prompt}
        ]
        output = chatbot(messages)
        response = output[0]['generated_text'][2]['content']
        answers.append(response)
    del chatbot
    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([polarity, model_name] + answers)




if __name__ == "__main__":
    # sys.exit(recipe_main())
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
        help="",
        required=True
    )

    args = parser.parse_args()
    models = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]


    with open(args.prompts, 'r') as file:
        count = sum(1 for line in file)


    csv_headers = ['polarity','model'] 
    for i in range(count):
        csv_headers.append(f'prompt_{i}')
    print("headers: ",csv_headers)


    
    with open(args.outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    count = 0
    for model in models:
        # tokenizer = AutoTokenizer.from_pretrained(model)
        # mod = AutoModelForCausalLM.from_pretrained(model)
        mod = ''
        if (args.counter == 0):
            for polarity in ['right', 'left']:
                one_response(model,mod,args.prompts,polarity, args.outfile)
        elif (args.counter == 1):
            # polarity = 'left'
            opener = args.prompts.split('/')[-1].split('_')[1]
            if opener == 'left':
                polarity = 'right'
            elif opener == 'right':
                polarity = 'left'
            # print(polarity)
            counter_arg(model, mod,args.prompts,polarity, args.outfile)


