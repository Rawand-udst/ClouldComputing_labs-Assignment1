import argparse
import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.train_data)

    # Keep only columns needed for tsfresh
    feature_cols = (
        ["unit_number", "time_in_cycles", "RUL",
         "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    df = df[feature_cols].copy()

    # Sort for time-series feature extraction
    df = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

    # Create target: max RUL per engine to align with one extracted row per engine
    target_df = df.groupby("unit_number")["RUL"].max().reset_index()

    # Remove target before tsfresh
    tsfresh_input = df.drop(columns=["RUL"])

    # Minimal feature extraction to keep output manageable
    features = extract_features(
        tsfresh_input,
        column_id="unit_number",
        column_sort="time_in_cycles",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0
    )

    impute(features)

    # Merge target back
    features = features.reset_index().rename(columns={"index": "unit_number"})
    final_df = features.merge(target_df, on="unit_number", how="inner")

    os.makedirs(args.out, exist_ok=True)
    final_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Extracted features shape:", final_df.shape)


if __name__ == "__main__":
    main()