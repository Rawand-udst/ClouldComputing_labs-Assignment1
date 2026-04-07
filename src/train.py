import argparse
from email import parser
import os
from pyexpat import model
import time
import gc

import azureml.mlflow
import mlflow
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# Columns to load — TF-IDF + SBERTskipped entirely (OOM)
# --------------------------------------------------
NEEDED_COLS = [
    "overall",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_compound",
    "review_length_chars",
    "review_length_words",
]


# --------------------------------------------------
# Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",     type=str,   required=True)
    parser.add_argument("--val_data",       type=str,   required=True)
    parser.add_argument("--test_data",      type=str,   required=True)
    parser.add_argument("--output",         type=str,   required=True)
    parser.add_argument("--max_train_rows", type=int, default=50000)
    parser.add_argument("--max_val_rows",   type=int, default=15000)
    parser.add_argument("--max_test_rows",  type=int, default=15000)
    parser.add_argument("--C",        type=float, default=0.4765063986113154)
    parser.add_argument("--max_iter", type=int,   default=300)
    return parser.parse_args()


# --------------------------------------------------
# Load — column pruning + row slicing at read time
# --------------------------------------------------
def load_dataset(path: str, max_rows: int = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    table = pq.read_table(path, columns=NEEDED_COLS)

    if max_rows is not None:
        table = table.slice(0, max_rows)

    df = table.to_pandas()
    print(f"Loaded {len(df)} rows | columns: {df.columns.tolist()}")
    return df


# --------------------------------------------------
# Labels
# --------------------------------------------------
def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' not found.")
    df = df[df["overall"].isin([1, 2, 4, 5])].copy()
    df["label"] = (df["overall"] >= 4).astype(np.int8)
    return df


# --------------------------------------------------
# Features — sentiment + length
# --------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feature_cols = [
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_neu",
        "sentiment_compound",
        "review_length_chars",
        "review_length_words",
    ]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    print(f"Feature matrix shape: {X.shape}")
    return X


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate(model, X: np.ndarray, y: np.ndarray, split_name: str):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    accuracy  = accuracy_score(y, y_pred)
    auc       = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall    = recall_score(y, y_pred, zero_division=0)
    f1        = f1_score(y, y_pred, zero_division=0)

    mlflow.log_metric(f"{split_name}_accuracy",  accuracy)
    mlflow.log_metric(f"{split_name}_auc",       auc)
    mlflow.log_metric(f"{split_name}_precision", precision)
    mlflow.log_metric(f"{split_name}_recall",    recall)
    mlflow.log_metric(f"{split_name}_f1",        f1)

    print(
        f"[{split_name}] acc={accuracy:.4f}  auc={auc:.4f}  "
        f"prec={precision:.4f}  rec={recall:.4f}  f1={f1:.4f}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    mlflow.start_run()

    # ---------- load ----------
    print("Loading train data...")
    train_df = load_dataset(args.train_data, max_rows=args.max_train_rows)

    print("Loading val data...")
    val_df = load_dataset(args.val_data, max_rows=args.max_val_rows)

    print("Loading test data...")
    test_df = load_dataset(args.test_data, max_rows=args.max_test_rows)

    # ---------- labels ----------
    print("Creating labels...")
    train_df = create_binary_labels(train_df)
    val_df   = create_binary_labels(val_df)
    test_df  = create_binary_labels(test_df)

    # ---------- features ----------
    print("Building train features...")
    X_train = build_feature_matrix(train_df)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    del train_df
    gc.collect()

    print("Building val features...")
    X_val = build_feature_matrix(val_df)
    y_val = val_df["label"].to_numpy(dtype=np.int8)
    del val_df
    gc.collect()

    print("Building test features...")
    X_test = build_feature_matrix(test_df)
    y_test = test_df["label"].to_numpy(dtype=np.int8)
    del test_df
    gc.collect()

    # ---------- log params ----------
    mlflow.log_param("model",             "LogisticRegression")
    mlflow.log_param("feature_type", "sentiment+length")
    mlflow.log_param("feature_dimension", int(X_train.shape[1]))
    mlflow.log_param("data_split",        "60_15_15_10")
    mlflow.log_param("C",                 args.C)
    mlflow.log_param("max_iter",          args.max_iter)
    mlflow.log_param("max_train_rows",    args.max_train_rows)
    mlflow.log_param("max_val_rows",      args.max_val_rows)
    mlflow.log_param("max_test_rows",     args.max_test_rows)

    # ---------- train ----------
    print("Training model...")
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver="liblinear",  
        random_state=42,     
    )
    model.fit(X_train, y_train)

    # ---------- evaluate ----------
    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    del X_train, y_train
    gc.collect()

    evaluate(model, X_val, y_val, "val")
    del X_val, y_val
    gc.collect()

    evaluate(model, X_test, y_test, "test")
    del X_test, y_test
    gc.collect()

    # ---------- save ----------
    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"Total runtime: {runtime:.1f}s")

    mlflow.end_run()
    print("Done.")


if __name__ == "__main__":
    main()
