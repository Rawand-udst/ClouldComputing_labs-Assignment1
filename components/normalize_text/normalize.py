import argparse
import os
import re
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="reviewText")
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Replace URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " URL ", text)

    # Replace numbers
    text = re.sub(r"\d+", " NUM ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Trim whitespace
    text = text.strip()

    return text

def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in input data")

    # Normalize
    df[args.text_col] = df[args.text_col].apply(normalize_text)

    # Remove empty or very short reviews (<10 characters)
    df = df[df[args.text_col].str.len() >= 10]

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Rows after filtering:", len(df))

if __name__ == "__main__":
    main()