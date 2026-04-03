import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def load_df(path: str) -> pd.DataFrame:
    parquet_path = os.path.join(path, "data.parquet")
    return pd.read_parquet(parquet_path)


def extract_helpful(x):
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return int(x[0]), int(x[1])
    return 0, 0


def main():
    args = parse_args()

    df = load_df(args.data)

    if "helpful" not in df.columns:
        raise ValueError("Column 'helpful' not found in input data")

    pairs = df["helpful"].apply(extract_helpful)

    out = df[["asin", "reviewerID"]].copy()
    out["helpful_votes"] = pairs.apply(lambda t: t[0]).astype(int)
    out["total_votes"] = pairs.apply(lambda t: t[1]).astype(int)
    out["helpful_ratio"] = out["helpful_votes"] / out["total_votes"].replace(0, 1)

    os.makedirs(args.out, exist_ok=True)
    out.to_parquet(os.path.join(args.out, "data.parquet"), index=False)


    print("Rows processed:", len(out))


if __name__ == "__main__":
    main()