import argparse
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--mi_sample_size", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    # Keep unit_number as identifier and RUL as target
    id_col = "unit_number"
    target_col = "RUL"

    X = df.drop(columns=[id_col, target_col])
    y = df[target_col]
    ids = df[[id_col]].copy()

    # 1) Variance Threshold
    vt = VarianceThreshold(threshold=0.0)
    X_vt = vt.fit_transform(X)
    selected_vt_cols = X.columns[vt.get_support()]
    X_vt = pd.DataFrame(X_vt, columns=selected_vt_cols, index=X.index)

    # 2) Correlation Filter
    corr_matrix = X_vt.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_corr = X_vt.drop(columns=to_drop)

    # 3) Mutual Information on a small sample to keep runtime manageable
    sample_n = min(args.mi_sample_size, len(X_corr))
    X_corr_sample = X_corr.sample(n=sample_n, random_state=42)
    y_sample = y.loc[X_corr_sample.index]

    mi_scores = mutual_info_regression(
        X_corr_sample,
        y_sample,
        random_state=42,
        n_jobs=-1
    )

    mi_series = pd.Series(mi_scores, index=X_corr.columns).sort_values(ascending=False)
    selected_mi_cols = mi_series.head(args.top_k).index.tolist()

    X_final = X_corr[selected_mi_cols].copy()

    # Save compact reduced dataset
    final_df = pd.concat([ids, X_final, y], axis=1)

    os.makedirs(args.out, exist_ok=True)
    final_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Original shape:", df.shape)
    print("After variance threshold:", X_vt.shape)
    print("After correlation filter:", X_corr.shape)
    print("After mutual information:", X_final.shape)
    print("Final saved shape:", final_df.shape)


if __name__ == "__main__":
    main()