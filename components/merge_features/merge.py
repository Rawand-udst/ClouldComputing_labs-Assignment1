# merge.py  ✅ Option A: memory-safe merge (drops giant columns like embeddings before saving)

import argparse
import os
import gc
import pandas as pd

KEYS = ["asin", "reviewerID"]

# text / metadata we never want in the merged feature set
DROP_COLS = [
    "reviewText", "summary", "reviewTime", "reviewerName",
    "title", "description"
]

# common names used for embedding/vector columns (drop to prevent OOM)
EMBED_COL_CANDIDATES = [
    "bert_embedding", "sbert_embedding", "semantic_embedding",
    "embedding", "embeddings", "vector", "sentence_embedding"
]

# azureml parquet outputs are usually out/data.parquet
def parquet_path(folder_path: str) -> str:
    return os.path.join(folder_path, "data.parquet")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--length", required=True)
    p.add_argument("--sentiment", required=True)
    p.add_argument("--tfidf", required=True)
    p.add_argument("--sbert", required=True)
    p.add_argument("--helpful", required=True)
    p.add_argument("--readability", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()

def read_df(folder_path: str) -> pd.DataFrame:
    path = parquet_path(folder_path)
    df = pd.read_parquet(path)

    # drop huge text columns early
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # ensure keys exist
    missing = [k for k in KEYS if k not in df.columns]
    if missing:
        raise ValueError(f"Missing key columns {missing} in {path}")

    # remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def merge_inner(left: pd.DataFrame, right: pd.DataFrame, tag: str) -> pd.DataFrame:
    merged = left.merge(right, on=KEYS, how="inner", sort=False, copy=False)
    del left, right
    gc.collect()
    print(f"✅ merged {tag}: rows={len(merged)} cols={len(merged.columns)}")
    return merged

def drop_big_vector_cols(df: pd.DataFrame) -> pd.DataFrame:
    dropped = []
    for c in EMBED_COL_CANDIDATES:
        if c in df.columns:
            df = df.drop(columns=[c])
            dropped.append(c)

    # also drop any column that is list/array-like stored as object (very common for embeddings)
    # (keeps keys safe)
    obj_cols = [c for c in df.columns if c not in KEYS and df[c].dtype == "object"]
    for c in obj_cols:
        # heuristic: if any value looks like a list/array, drop it
        try:
            sample = df[c].dropna().head(3).tolist()
            if any(isinstance(x, (list, tuple)) for x in sample):
                df = df.drop(columns=[c])
                dropped.append(c)
        except Exception:
            pass

    if dropped:
        print("🧹 Dropped large vector/object cols:", dropped)
    else:
        print("🧹 No embedding/vector cols detected to drop.")

    return df

def main():
    args = parse_args()

    # load smaller first
    df = read_df(args.length)
    df = merge_inner(df, read_df(args.sentiment), "sentiment")
    df = merge_inner(df, read_df(args.helpful), "helpful")
    df = merge_inner(df, read_df(args.readability), "readability")

    # big ones later
    df = merge_inner(df, read_df(args.tfidf), "tfidf")
    df = merge_inner(df, read_df(args.sbert), "sbert")

    # ✅ Option A core: drop embeddings/vectors before writing final merged parquet
    df = drop_big_vector_cols(df)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(out_path, index=False)

    print("✅ FINAL rows:", len(df))
    print("✅ FINAL cols:", len(df.columns))
    print("✅ Saved to:", out_path)

if __name__ == "__main__":
    main()