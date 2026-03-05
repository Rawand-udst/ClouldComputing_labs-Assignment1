import argparse
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--val", type=str, required=True)
    p.add_argument("--test", type=str, required=True)

    p.add_argument("--text_col", type=str, default="reviewText")

    p.add_argument("--max_features", type=int, default=5000)
    p.add_argument("--ngram_min", type=int, default=1)
    p.add_argument("--ngram_max", type=int, default=2)

    p.add_argument("--out_train", type=str, required=True)
    p.add_argument("--out_val", type=str, required=True)
    p.add_argument("--out_test", type=str, required=True)
    p.add_argument("--out_model", type=str, required=True)
    return p.parse_args()

def load_df(folder_path: str) -> pd.DataFrame:
    # Azure ML uri_folder will contain parquet(s) - in our previous components we used data.parquet
    p = os.path.join(folder_path, "data.parquet")
    if not os.path.exists(p):
        # fallback: read any parquet file in folder
        files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
        if not files:
            raise FileNotFoundError(f"No parquet found in {folder_path}")
        p = os.path.join(folder_path, files[0])
    return pd.read_parquet(p)

def save_df(df: pd.DataFrame, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    df.to_parquet(os.path.join(out_folder, "data.parquet"), index=False)

def main():
    args = parse_args()

    train_df = load_df(args.train)
    val_df = load_df(args.val)
    test_df = load_df(args.test)

    for d in [train_df, val_df, test_df]:
        if args.text_col not in d.columns:
            raise ValueError(f"Column '{args.text_col}' not found in input data")
        d[args.text_col] = d[args.text_col].fillna("").astype(str)

    # ✅ Fit ONLY on TRAIN
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1,2)
    )

    X_train = vectorizer.fit_transform(train_df[args.text_col])
    X_val = vectorizer.transform(val_df[args.text_col])
    X_test = vectorizer.transform(test_df[args.text_col])

    # Store as list of (index,value) pairs to keep it parquet-friendly
    def sparse_to_pairs(X):
        pairs = []
        for i in range(X.shape[0]):
            row = X.getrow(i)
            pairs.append(list(zip(row.indices.tolist(), row.data.tolist())))
        return pairs

    train_df["tfidf_features"] = sparse_to_pairs(X_train)
    val_df["tfidf_features"] = sparse_to_pairs(X_val)
    test_df["tfidf_features"] = sparse_to_pairs(X_test)

    save_df(train_df, args.out_train)
    save_df(val_df, args.out_val)
    save_df(test_df, args.out_test)

    # Save the fitted vectorizer (so it’s clear we fit on train only)
    os.makedirs(args.out_model, exist_ok=True)
    with open(os.path.join(args.out_model, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("Done TF-IDF ✅")
    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Test rows:", len(test_df))

if __name__ == "__main__":
    main()