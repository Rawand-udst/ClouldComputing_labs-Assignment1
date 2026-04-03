# Couldn't use it cuz i got out of memory error when i ran the pipleline so i had to remove it so sad
import argparse
import os
import re
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def load_df(path: str) -> pd.DataFrame:
    parquet_path = os.path.join(path, "data.parquet")
    return pd.read_parquet(parquet_path)


def main():
    args = parse_args()

    df = load_df(args.data)

    if "reviewText" not in df.columns:
        raise ValueError("Column 'reviewText' not found in input data")

    text = df["reviewText"].fillna("").astype(str)

    words = text.apply(lambda s: re.findall(r"\b\w+\b", s))
    word_count = words.apply(len)
    char_count = text.str.len()
    avg_word_len = words.apply(lambda ws: (sum(len(w) for w in ws) / len(ws)) if len(ws) else 0.0)
    sentence_count = text.apply(lambda s: len([x for x in re.split(r"[.!?]+", s) if x.strip()]))
    avg_sentence_len_words = (word_count / sentence_count.replace(0, 1)).astype(float)

    out = df[["asin", "reviewerID"]].copy()
    out["word_count"] = word_count.astype(int)
    out["char_count"] = char_count.astype(int)
    out["avg_word_len"] = avg_word_len.astype(float)
    out["sentence_count"] = sentence_count.astype(int)
    out["avg_sentence_len_words"] = avg_sentence_len_words.astype(float)

    os.makedirs(args.out, exist_ok=True)
    out.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Rows processed:", len(out))


if __name__ == "__main__":
    main()