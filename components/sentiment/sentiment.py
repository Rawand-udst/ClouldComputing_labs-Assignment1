import argparse
import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    
    nltk.data.path.append("/tmp/nltk_data")
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", download_dir="/tmp/nltk_data")

    args = parse_args()

    df = pd.read_parquet(args.data)
    sia = SentimentIntensityAnalyzer()

    scores = df["reviewText"].astype(str).apply(sia.polarity_scores)

    df["sentiment_pos"] = scores.apply(lambda x: x["pos"])
    df["sentiment_neg"] = scores.apply(lambda x: x["neg"])
    df["sentiment_neu"] = scores.apply(lambda x: x["neu"])
    df["sentiment_compound"] = scores.apply(lambda x: x["compound"])

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Rows processed:", len(df))


if __name__ == "__main__":
    main()
