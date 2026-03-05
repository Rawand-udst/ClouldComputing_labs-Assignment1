import argparse, os, re
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()

def load_df(folder):
    return pd.read_parquet(os.path.join(folder, "data.parquet"))

def main():
    args = parse_args()
    df = load_df(args.data)

    text = df["reviewText"].fillna("").astype(str)

    # tokens / counts
    words = text.apply(lambda s: re.findall(r"\b\w+\b", s))
    word_count = words.apply(len)
    char_count = text.str.len()

    avg_word_len = words.apply(lambda ws: (sum(len(w) for w in ws) / len(ws)) if len(ws) else 0.0)

    # sentence count (simple)
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

    print("Rows:", len(out))
    print(out.head())

if __name__ == "__main__":
    main()