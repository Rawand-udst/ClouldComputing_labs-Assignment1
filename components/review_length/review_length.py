import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="reviewText")
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in input data")

    # Fill nulls just in case
    df[args.text_col] = df[args.text_col].fillna("").astype(str)

    # Features
    df["review_length_chars"] = df[args.text_col].str.len()
    df["review_length_words"] = df[args.text_col].str.split().apply(len)

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Rows:", len(df))
    print("Added features: review_length_chars, review_length_words")

if __name__ == "__main__":
    main()