import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def trend_features(group, sensor_name):
    first_val = group[sensor_name].iloc[0]
    last_val = group[sensor_name].iloc[-1]
    return first_val, last_val, last_val - first_val


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)
    df = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

    rows = []

    for unit_number, group in df.groupby("unit_number"):
        s11_first, s11_last, s11_delta = trend_features(group, "sensor_11")
        s12_first, s12_last, s12_delta = trend_features(group, "sensor_12")
        s15_first, s15_last, s15_delta = trend_features(group, "sensor_15")

        rows.append({
            "unit_number": unit_number,
            "sensor_11_first": s11_first,
            "sensor_11_last": s11_last,
            "sensor_11_delta": s11_delta,
            "sensor_12_first": s12_first,
            "sensor_12_last": s12_last,
            "sensor_12_delta": s12_delta,
            "sensor_15_first": s15_first,
            "sensor_15_last": s15_last,
            "sensor_15_delta": s15_delta,
            "RUL": group["RUL"].max()
        })

    feature_df = pd.DataFrame(rows)

    os.makedirs(args.out, exist_ok=True)
    feature_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Trend features shape:", feature_df.shape)


if __name__ == "__main__":
    main()