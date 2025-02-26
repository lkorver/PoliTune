import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import argparse
import json
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files for scoring.")
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        required=True,
        help="The path to the intput file.",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        required=True,
        help="The path to the output file.",
    )
    parser.add_argument(
        "--model", 
        "-M",
        type=str,
        required=True,
        help="The model name on Hugging Face "
    )
    parser.add_argument(
        "--index", 
        type=str,
        help="Index of which prompts to analyze"
    )
    parser.add_argument("--start-at", type=int, default=0)
    args = parser.parse_args()

    count = 25

    model_name = args.model
    # tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    csv_headers = ['model', 'position'] 
    for i in range(count):
        csv_headers.append(f'prompt_{i}')

    with open(args.input, 'r') as file:
        data = json.load(file)
        answers = []
        for row in data:
            rd = {}
            rd['model'] = row['model']
            rd['position'] = row['position']
            for i in range(count):
                
                resp = row[f'prompt_{i}']
                inputs = tokenizer(resp, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_class_id = logits.argmax().item()
                rd[f'prompt_{i}'] = logits.tolist()
            answers.append(rd)
    with open(args.output, 'w') as json_file:
        json.dump(answers, json_file, indent=4)

            

