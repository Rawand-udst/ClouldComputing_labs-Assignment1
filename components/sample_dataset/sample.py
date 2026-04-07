import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load all parquet shards in folder
    df = pd.read_parquet(args.data)

    if len(df) > args.n_rows:
        df = df.sample(n=args.n_rows, random_state=args.seed)

    os.makedirs(args.out, exist_ok=True)

    # Write a single parquet file for downstream components
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Rows after sampling:", len(df))


if __name__ == "__main__":
    main()