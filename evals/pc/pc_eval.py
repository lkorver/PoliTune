import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process CSV files for scoring.")
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        required=True,
        help="The path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        required=True,
        help="The path to the output CSV file.",
    )
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    print(f"Loaded {args.input} into DataFrame.")

    column_names = ["iteration","step"]
    for i in range(0,62):
        column_names.append(f'answer_{i}')

    out_df = pd.DataFrame(columns=column_names)

    for i in range(df.shape[0]):
        new_arr = []
        for j in range(0,62):
            p0 = df[f'question_{j}'][i]
            if (p0.lower().find("strongly agree") != -1) or (p0.lower().find("3") != -1):
                new_arr.append(3)
            elif (p0.lower().find("strongly disagree") != -1) or (p0.lower().find("0") != -1):
                new_arr.append(0)
            elif (p0.lower().find("disagree") != -1) or (p0.lower().find("1") != -1):
                new_arr.append(1)
            elif (p0.lower().find("agree") != -1) or (p0.lower().find("2") != -1):
                new_arr.append(2)
            else:
                new_arr.append(-1)
        new_row = {f'answer_{i}': new_arr[i] for i in range(0, 62)}
        # print("new row",new_row)
        new_row['iteration'] = df['iteration'][i]
        new_row['step'] = df['step'][i]
        out_df.loc[len(out_df)] = new_row

    out_df.to_csv(args.output)
    print(f"Saved results to {args.output}.")


if __name__ == "__main__":
    main()
