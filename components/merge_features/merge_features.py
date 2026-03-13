import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_features", type=str, required=True)
    parser.add_argument("--summary_features", type=str, required=True)
    parser.add_argument("--trend_features", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def drop_duplicate_target(df, target_col="RUL", id_col="unit_number"):
    cols_to_keep = [c for c in df.columns if c != target_col or c == id_col]
    return df[cols_to_keep]


def main():
    args = parse_args()

    filtered_df = pd.read_parquet(args.filtered_features)
    summary_df = pd.read_parquet(args.summary_features)
    trend_df = pd.read_parquet(args.trend_features)

    # Keep target only from filtered_df
    summary_df = summary_df.drop(columns=["RUL"], errors="ignore")
    trend_df = trend_df.drop(columns=["RUL"], errors="ignore")

    merged_df = filtered_df.merge(summary_df, on="unit_number", how="inner")
    merged_df = merged_df.merge(trend_df, on="unit_number", how="inner")

    os.makedirs(args.out, exist_ok=True)
    merged_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Merged feature table shape:", merged_df.shape)
    print("Columns:", list(merged_df.columns))


if __name__ == "__main__":
    main()