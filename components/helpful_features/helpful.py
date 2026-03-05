import argparse, os
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

    # helpful is array like [helpful_votes, total_votes]
    def hv(x):
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            return int(x[0]), int(x[1])
        return 0, 0

    pairs = df["helpful"].apply(hv)
    out = df[["asin", "reviewerID"]].copy()
    out["helpful_votes"] = pairs.apply(lambda t: t[0])
    out["total_votes"] = pairs.apply(lambda t: t[1])
    out["helpful_ratio"] = out["helpful_votes"] / (out["total_votes"].replace(0, 1))

    os.makedirs(args.out, exist_ok=True)
    out.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Rows:", len(out))
    print(out.head())

if __name__ == "__main__":
    main()