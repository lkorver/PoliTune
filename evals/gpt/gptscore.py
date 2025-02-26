import csv
import argparse
import os
import sys
from tqdm import tqdm
from gptscore_utils import process_prompt, truncate_to_n_words, count_rows, spacy_eval
import time
import openai
from openai import OpenAI


def getScoreFrom20(client, prompts, key, model, numerical):
    return process_prompt(
        client,
        prompts,
        key,
        model,
        numerical,
        max_tokens=100,
        temperature=0,
        min_score=0,
        max_score=20,
        no_progress=True,
        quiet=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Process CSV files for scoring.")
    parser.add_argument("--key", type=str, help="OpenAI API key")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        # default="gpt-4",
        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"],
        help="The model to use.",
    )
    parser.add_argument(
        "--prompts",
        "-P",
        type=str,
        required=True,
        help="The path to the prompts CSV file.",
    )
    parser.add_argument(
        "--responses",
        "-R",
        type=str,
        required=True,
        help="The path to the responses CSV file.",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        required=True,
        help="The path for the new CSV output file.",
    )
    parser.add_argument(
        "--indices",
        "-I",
        type=str,
        required=True,
        help="The number ranges in different formats.",
    )
    parser.add_argument(
        "--spacy",
        type=int,
        required=True,
        help="1 if you want spacy evals, 0 otherwise",
    )
    parser.add_argument(
        "--numerical",
        type=int,
        required=True,
        help="1 if you want only a numerical value score (will default to middle range if no number is given)."
    )
    parser.add_argument("--start-at", type=int, default=0)

    args = parser.parse_args()

    key = args.key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Please provide a OpenAI API key.")
        sys.exit(1)
    client = OpenAI(api_key=key)

    word_count = 0
    first_request = True
    start_time = time.time()

    ranges = args.indices.split(",")
    valid_indices = []
    for r in ranges:
        if "-" in r:
            start, end = map(int, r.split("-"))
            valid_indices.extend(range(start, end + 1))
        else:
            valid_indices.append(int(r))
    with open(args.prompts, "r") as prompts_f:
        reader = csv.reader(prompts_f)
        header = next(reader)
        for index, row in enumerate(reader):
            if index in valid_indices:
                print(f"Including prompt {row[0]}: {row[1]}")

    avg_score_20 = []
    individual_scores_20 = {}
    scores_spacy = []

    failed = False
    num_responses = count_rows(args.responses)
    with open(args.responses, "r") as responses_f, open(
        args.output, "a", newline=""
    ) as output_f:
        reader = csv.reader(responses_f)
        writer = csv.writer(output_f)
        header = next(reader)
        new_header = []

        for index, col_name in enumerate(header[2:]):
            if index in valid_indices:
                new_header.extend([f"{col_name}_score20"])
                individual_scores_20[col_name] = []

        new_header.extend(["average_score20"])
        
        if (args.spacy == 1):
            new_header = ["polarity","subjectivity","assessments"]

        if os.stat(args.output).st_size == 0 or args.start_at == 0:
            output_f.truncate(0)
            writer.writerow(new_header)

        if args.start_at > 0:
            for _ in range(args.start_at + 1):
                next(reader)

        try:
            for row_idx, row in enumerate(
                tqdm(
                    reader,
                    initial=args.start_at,
                    total=num_responses,
                    desc="Processing responses",
                ),
                start=args.start_at,
            ):
                print(row_idx)
                new_row = []
                total_score_20 = 0
                count = 0
                prompt_list_20 = []

                for index, prompt in enumerate(row[2:]):
                    if index in valid_indices:
                        truncated_prompt = truncate_to_n_words(prompt, 200)
                        # add the words from the system prompt
                        word_count += len(truncated_prompt.split()) + 70
                        print(truncated_prompt)
                        prompt_list_20.append(truncated_prompt)

                elapsed_time = time.time() - start_time
                if not first_request and word_count / (elapsed_time * 60) > 8000:
                    delay = 60 - (elapsed_time % 60)
                    print(
                        f"Rate limit exceeded, current rate {word_count / (elapsed_time * 60)} word/min. Waiting {delay} seconds."
                    )
                    time.sleep(delay)
                    start_time = time.time()
                    word_count = 0
                first_request = False

                # spacy eval
                if (args.spacy == 1):
                    polarity, subjectivity, assessments = spacy_eval(prompt_list_20,)
                    new_row.extend([polarity,subjectivity,assessments])
                    writer.writerow(new_row)
                    output_f.flush()
                else:
                    # Obtain the scores by processing all the prompts at once
                    scores_20 = getScoreFrom20(client, prompt_list_20, key, args.model, args.numerical)
                    print("**********",scores_20)
                    for index, (score_20) in enumerate(
                        zip(scores_20)
                    ):
                        col_name = header[2 + valid_indices[index]]
                        new_row.extend([score_20])
                        individual_scores_20[col_name].append(score_20)
                        score_20 = score_20[0]
                        if (args.numerical == 1):
                            total_score_20 += score_20
                        count += 1

                    if (args.numerical == 1):
                        avg_20 = total_score_20 / count if count > 0 else 0
                        avg_score_20.append(avg_20)

                        new_row.extend([avg_20])
                    writer.writerow(new_row)
                    output_f.flush()
        except openai.RateLimitError as e:
            print(
                f"{e}\nRate limit exceeded at row {row_idx}. Estimated rate {word_count / (elapsed_time * 60)} word/min."
            )
            failed = True
        except Exception as e:
            print(
                f"An error occurred: {e}.\nLast processed row index: {row_idx}. You can resume from this index."
            )
            failed = True


if __name__ == "__main__":
    main()
