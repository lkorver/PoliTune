import csv
import argparse

from sentence_transformers import SentenceTransformer

def by_sentence(infile, outfile, model_name):

    model = SentenceTransformer(model_name)
    csv_headers = ['iteration','step'] 
    with open(infile, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(f)
        print(headers)
        for col in headers[:-2].split(',')[2:]:
            csv_headers.append(col)
        print(csv_headers)

        with open(outfile, 'w', newline='') as out:
            writer = csv.writer(out, delimiter=',')
            writer.writerow(csv_headers)
        num_prompts = len(headers.split(',')) - 2
        
        for row_index, row in enumerate(reader):
            data = row[:2]
            for i,response in enumerate(row[2:]):
                embeddings = []
                for sentence in response.split('.'):
                    embedding = model.encode(sentence)
                    embeddings.append(embedding)
                data.append(embeddings)
            with open(outfile, 'a', encoding='utf-8') as out:
                writer = csv.writer(out)
                writer.writerow(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outfile",
        type=str,
        help="The output csv file.",
        required=True
    )
    parser.add_argument(
        "--infile",
        type=str,
        help="The csv file containing the data",
        required=True
    )
    args = parser.parse_args()

    model = "all-mpnet-base-v2"
    by_sentence(args.infile,args.outfile,model)

    