import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
# import spacytextblob.spacytextblob as tb
import csv
import os
import argparse
from tqdm import tqdm
import time
from gptscore_utils import count_rows


def spacy_eval(prompts):

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    answers = []
    for prompt in prompts:
        scoring = {}
        doc = nlp(prompt)

        scoring['polarity'] = doc._.blob.polarity
        scoring['subjectivity'] = doc._.blob.subjectivity
        scoring['assessments'] = doc._.blob.sentiment_assessments.assessments
        answers.append(scoring)
    return answers

def main():
    parser = argparse.ArgumentParser(description="Process CSV files for scoring.")
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
    parser.add_argument("--start-at", type=int, default=0)

    args = parser.parse_args()


    word_count = 0
    first_request = True
    start_time = time.time()

    scores_spacy = []

    num_responses = count_rows(args.responses)
    with open(args.responses, "r") as responses_f, open(
        args.output, "a", newline=""
    ) as output_f:
        reader = csv.reader(responses_f)
        writer = csv.writer(output_f)
        header = next(reader)
        new_header = header

        new_header.extend(["average_spacy"])
        print(new_header)
        scoring = {}

        if os.stat(args.output).st_size == 0 or args.start_at == 0:
            output_f.truncate(0)
            writer.writerow(new_header)

        if args.start_at > 0:
            for _ in range(args.start_at + 1):
                next(reader)


        for row_idx, row in enumerate(
            tqdm(
                reader,
                initial=args.start_at,
                total=num_responses,
                desc="Processing responses",
            ),
            start=args.start_at,
        ):
            new_row = row[:2]
            total_score_20 = 0
            count = 0
            prompt_list = []

            for index, prompt in enumerate(row[2:]):
                prompt_list.append(prompt)


            # spacy eval
            answers = spacy_eval(prompt_list)
            pol, sub = 0, 0
            count = 0
            for ans in answers:
                pol += abs(ans['polarity'])
                sub += ans['subjectivity']
                count += 1
            average_spacy = {}
            average_spacy['polarity'] = pol/count
            average_spacy['subjectivity'] = sub/count
            answers.append(average_spacy)
            new_row.extend(answers)
            writer.writerow(new_row)
            output_f.flush()


if __name__ == "__main__":
    main()
