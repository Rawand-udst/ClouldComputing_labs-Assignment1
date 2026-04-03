import argparse
import os
import pandas as pd
import re



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """
    Normalize review text by:
    - lowercasing
    - replacing URLs with <URL>
    - replacing numbers with <NUM>
    - replacing emojis with <EMOJI>
    - removing punctuation
    - collapsing whitespace
    """

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # replace URLs
    text = re.sub(r"http\S+|www\S+", "<URL>", text)

    # replace numbers
    text = re.sub(r"\d+", "<NUM>", text)

    # replace emojis
    text = text.encode("ascii", "ignore").decode("ascii")

    # remove punctuation (keep placeholders)
    text = re.sub(r"[^\w\s<>]", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    args = parse_args()

    # Load data
    df = pd.read_parquet(args.data)

    # Normalize text
    df["reviewText"] = df["reviewText"].apply(normalize_text)

    # Filter out empty or very short reviews (<10 characters)
    df = df[df["reviewText"].str.len() >= 10]

    # Write output
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Rows after normalization:", len(df))


if __name__ == "__main__":
    main()
