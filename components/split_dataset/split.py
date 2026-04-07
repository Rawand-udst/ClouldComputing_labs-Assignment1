import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    df = pd.read_parquet(args.data)

    if "review_year" not in df.columns:
        raise ValueError("Input data must contain a 'review_year' column for the deployment split.")

    # Sort newest reviews are last
    df = df.sort_values("review_year", ascending=True).reset_index(drop=True)

    # Deployment split = newest 10% of full dataset
    n_deploy = int(len(df) * 0.10)

    if n_deploy > 0:
        deploy_df = df.tail(n_deploy)
        remaining_df = df.iloc[:-n_deploy]
    else:
        deploy_df = df.iloc[0:0].copy()
        remaining_df = df.copy()

    # Split remaining 90% into train / val / test
    # Final target proportions over full dataset:
    # train = 60%, val = 15%, test = 15%, deploy = 10%
    # remaining 90%:
    # train = 60/90 = 2/3
    # val = 15/90 = 1/6
    # test = 15/90 = 1/6

    train_df, temp_df = train_test_split(
        remaining_df,
        test_size=1 / 3,
        random_state=args.seed,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=args.seed,
        shuffle=True
    )

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deploy_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)
    deploy_df.to_parquet(os.path.join(args.deploy_out, "data.parquet"), index=False)

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print("Deploy rows:", len(deploy_df))


if __name__ == "__main__":
    main()