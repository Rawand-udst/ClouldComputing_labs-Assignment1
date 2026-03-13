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
    parser.add_argument("--sample_engines", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_column_names():
    return (
        ["unit_number", "time_in_cycles"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )


def load_txt_file(path: str, column_names=None):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")

    if column_names is not None:
        df.columns = column_names

    return df


def sample_train_engines(train_df: pd.DataFrame, sample_engines: int, seed: int):
    unique_engines = sorted(train_df["unit_number"].unique())

    if sample_engines >= len(unique_engines):
        return train_df

    sampled = (
        pd.Series(unique_engines)
        .sample(n=sample_engines, random_state=seed)
        .sort_values()
        .tolist()
    )

    return train_df[train_df["unit_number"].isin(sampled)].copy()


def main():
    args = parse_args()

    columns = get_column_names()

    train_df = load_txt_file(args.train_data, columns)
    test_df = load_txt_file(args.test_data, columns)
    rul_df = load_txt_file(args.rul_data, ["RUL"])

    # Sample only the training engines to keep later feature extraction and merge manageable
    train_df = sample_train_engines(train_df, args.sample_engines, args.seed)

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.rul_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)
    rul_df.to_parquet(os.path.join(args.rul_out, "data.parquet"), index=False)

    print("Sampled train shape:", train_df.shape)
    print("Unique train engines:", train_df["unit_number"].nunique())
    print("Test shape:", test_df.shape)
    print("RUL shape:", rul_df.shape)


if __name__ == "__main__":
    main()