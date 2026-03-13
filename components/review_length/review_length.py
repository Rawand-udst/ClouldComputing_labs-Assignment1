import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    # Small handcrafted per-engine summary features
    grouped = df.groupby("unit_number")

    feature_df = pd.DataFrame({
        "unit_number": grouped.size().index,
        "engine_life_cycles": grouped["time_in_cycles"].max().values,
        "sensor_11_mean": grouped["sensor_11"].mean().values,
        "sensor_11_std": grouped["sensor_11"].std().fillna(0).values,
        "sensor_12_mean": grouped["sensor_12"].mean().values,
        "sensor_12_std": grouped["sensor_12"].std().fillna(0).values,
        "sensor_15_mean": grouped["sensor_15"].mean().values,
        "sensor_15_std": grouped["sensor_15"].std().fillna(0).values,
        "op_setting_1_mean": grouped["op_setting_1"].mean().values,
        "op_setting_2_mean": grouped["op_setting_2"].mean().values,
        "op_setting_3_mean": grouped["op_setting_3"].mean().values,
        "RUL": grouped["RUL"].max().values
    })

    os.makedirs(args.out, exist_ok=True)
    feature_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Review-length-style features shape:", feature_df.shape)


if __name__ == "__main__":
    main()