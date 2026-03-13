import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--rul_data", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--rul_out", type=str, required=True)
    return parser.parse_args()


def add_rul_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = train_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]

    train_df = train_df.merge(max_cycles, on="unit_number", how="left")
    train_df["RUL"] = train_df["max_cycle"] - train_df["time_in_cycles"]

    # Keep max_cycle only if you want debugging; otherwise drop it to keep output smaller
    train_df = train_df.drop(columns=["max_cycle"])
    return train_df


def main():
    args = parse_args()

    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)
    rul_df = pd.read_parquet(args.rul_data)

    train_rul_df = add_rul_to_train(train_df)

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.rul_out, exist_ok=True)

    train_rul_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)
    rul_df.to_parquet(os.path.join(args.rul_out, "data.parquet"), index=False)

    print("Train with RUL shape:", train_rul_df.shape)
    print("Train engines:", train_rul_df["unit_number"].nunique())
    print("Test shape:", test_df.shape)
    print("RUL shape:", rul_df.shape)


if __name__ == "__main__":
    main()