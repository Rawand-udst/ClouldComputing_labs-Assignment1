import argparse
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--text_col", type=str, default="reviewText")
    p.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def load_df(folder_path: str) -> pd.DataFrame:
    p = os.path.join(folder_path, "data.parquet")
    if not os.path.exists(p):
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

    df = load_df(args.data)

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found")

    texts = df[args.text_col].fillna("").astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = SentenceTransformer(args.model_name, device=device)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # ✅ reduce memory usage
    embeddings = embeddings.astype(np.float16)

    print("Embedding shape:", embeddings.shape)

    # keep only keys + embeddings
    keep_cols = ["asin", "reviewerID"]
    df = df[keep_cols]

    # create numeric embedding columns
    emb_dim = embeddings.shape[1]

    for i in range(emb_dim):
        df[f"emb_{i}"] = embeddings[:, i]

    save_df(df, args.out)

    print("Embedding done ✅")
    print("Rows:", len(df))
    print("Embedding dim:", emb_dim)


if __name__ == "__main__":
    main()