from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.fsspec import AzureMachineLearningFileSystem
import pandas as pd
import json
import requests
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# Azure ML workspace connection
# --------------------------------------------------
subscription_id = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
resource_group = "rg-60304948"
workspace_name = "Amazon-Electronics-Lab-60304948"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name
)

# --------------------------------------------------
# Load deploy dataset from Azure ML Data Asset
# --------------------------------------------------
data_asset = ml_client.data.get(
    name="amazon_review_merged_features_deploy",
    version="2"
)

deploy_path = data_asset.path.rstrip("/")
print("Deploy asset path:", deploy_path)

fs = AzureMachineLearningFileSystem(deploy_path)

matches = fs.glob("**/*.parquet")
print("Parquet matches:", matches)

if not matches:
    raise RuntimeError("No parquet files found in deploy asset.")

parquet_file = matches[0]

with fs.open(parquet_file, "rb") as f:
    df = pd.read_parquet(f)

print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# Create binary labels (same rule as training)
# --------------------------------------------------
if "overall" not in df.columns:
    raise RuntimeError("Column 'overall' is missing from deploy dataset.")

df = df[df["overall"].isin([1, 2, 4, 5])].copy()
df["label"] = (df["overall"] >= 4).astype(int)

# test on a small sample first so endpoint does not timeout
df = df.head(40000)

y_true = df["label"].values

# --------------------------------------------------
# Feature matrix (same as current train.py)
# sentiment + length only
# --------------------------------------------------
def build_feature_matrix(df):
    required_cols = [
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_neu",
        "sentiment_compound",
        "review_length_chars",
        "review_length_words",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")

    sentiment_features = df[
        ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]
    ].to_numpy(dtype=np.float32)

    length_features = df[
        ["review_length_chars", "review_length_words"]
    ].to_numpy(dtype=np.float32)

    return np.hstack([
        sentiment_features,
        length_features
    ]).astype(np.float32)

X = build_feature_matrix(df)

# --------------------------------------------------
# Endpoint information
# --------------------------------------------------
ENDPOINT_URL = "https://amazon-review-endpoint-60304948.qatarcentral.inference.ml.azure.com/score"
API_KEY = "9CbfOEGKBTwJjvt3oOpVcl6qqxJoDWP6doDAg3GbWzbZlWNtw7X4JQQJ99CDAAAAAAAAAAAAINFRAZML1kAt"
 

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# --------------------------------------------------
# Quick test request
# --------------------------------------------------
test_payload = {"data": X[:5].tolist()}

test_response = requests.post(
    ENDPOINT_URL,
    headers=headers,
    data=json.dumps(test_payload),
    timeout=60
)

print("Test status:", test_response.status_code)
print("Test response:", test_response.text)

if test_response.status_code != 200:
    raise RuntimeError("Test request failed. Fix endpoint first.")

# --------------------------------------------------
# Send requests in batches
# --------------------------------------------------
batch_size = 10
predictions = []

for i in range(0, len(X), batch_size):
    batch = X[i:i + batch_size]

    payload = {"data": batch.tolist()}

    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Request failed: {response.status_code} | {response.text}")

    result = response.json()

    if "predictions" not in result:
        raise RuntimeError(f"Unexpected response format: {result}")

    predictions.extend(result["predictions"])

# --------------------------------------------------
# Compute deployment metrics
# --------------------------------------------------
y_pred = np.array(predictions[:len(y_true)])

deploy_accuracy = accuracy_score(y_true[:len(y_pred)], y_pred)
deploy_precision = precision_score(y_true[:len(y_pred)], y_pred, zero_division=0)
deploy_recall = recall_score(y_true[:len(y_pred)], y_pred, zero_division=0)
deploy_f1 = f1_score(y_true[:len(y_pred)], y_pred, zero_division=0)

print("Deployment accuracy :", deploy_accuracy)
print("Deployment precision:", deploy_precision)
print("Deployment recall   :", deploy_recall)
print("Deployment f1       :", deploy_f1)
print("Predictions returned:", len(y_pred))