import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--deploy", type=str, required=True)

    parser.add_argument("--out_train", type=str, required=True)
    parser.add_argument("--out_val", type=str, required=True)
    parser.add_argument("--out_test", type=str, required=True)
    parser.add_argument("--out_deploy", type=str, required=True)

    parser.add_argument("--max_features", type=int, default=500)

    return parser.parse_args()


def add_tfidf_column(df, vectorizer):
    X = vectorizer.transform(df["reviewText"].astype(str))
    df = df.copy()
    df["tfidf_vector"] = X.toarray().tolist()  # fixed-length vector
    return df


def main():
    args = parse_args()

    train_df  = pd.read_parquet(os.path.join(args.train, "data.parquet"))
    val_df    = pd.read_parquet(os.path.join(args.val, "data.parquet"))
    test_df   = pd.read_parquet(os.path.join(args.test, "data.parquet"))
    deploy_df = pd.read_parquet(os.path.join(args.deploy, "data.parquet"))

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2)
    )

    # ------------------------------------------------
    # FIT ONLY ON TRAIN
    # ------------------------------------------------
    vectorizer.fit(train_df["reviewText"].astype(str))

    train_df  = add_tfidf_column(train_df, vectorizer)
    val_df    = add_tfidf_column(val_df, vectorizer)
    test_df   = add_tfidf_column(test_df, vectorizer)
    deploy_df = add_tfidf_column(deploy_df, vectorizer)

    os.makedirs(args.out_train, exist_ok=True)
    os.makedirs(args.out_val, exist_ok=True)
    os.makedirs(args.out_test, exist_ok=True)
    os.makedirs(args.out_deploy, exist_ok=True)

    train_df.to_parquet(os.path.join(args.out_train, "data.parquet"))
    val_df.to_parquet(os.path.join(args.out_val, "data.parquet"))
    test_df.to_parquet(os.path.join(args.out_test, "data.parquet"))
    deploy_df.to_parquet(os.path.join(args.out_deploy, "data.parquet"))

    print("TF-IDF complete")
    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print("Deploy rows:", len(deploy_df))
    print("Vocabulary size:", len(vectorizer.vocabulary_))


if __name__ == "__main__":
    main()