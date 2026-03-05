import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

    df[args.text_col] = df[args.text_col].fillna("").astype(str)

    # Make sure VADER lexicon is available
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    # Compute sentiment scores
    scores = df[args.text_col].apply(sia.polarity_scores)

    df["sentiment_pos"] = scores.apply(lambda x: x["pos"])
    df["sentiment_neg"] = scores.apply(lambda x: x["neg"])
    df["sentiment_neu"] = scores.apply(lambda x: x["neu"])
    df["sentiment_compound"] = scores.apply(lambda x: x["compound"])

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Rows:", len(df))
    print("Added features: sentiment_pos, sentiment_neg, sentiment_neu, sentiment_compound")

if __name__ == "__main__":
    main()