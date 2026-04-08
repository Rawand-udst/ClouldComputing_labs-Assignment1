# Assignment 2 — Model Training & Automation with Azure
**Name:** Rawand Elraba  
**Student ID:** 60304948  
**Branch:** `assignment2_model_training`  
**Course:** DSAI3202 — Winter 2026

---

## Overview

This assignment transitions the Amazon Electronics review project from feature engineering into the training and experimentation phase. Starting from the merged feature datasets produced in Lab 4, this assignment covers model training on Azure ML compute, hyperparameter tuning via sweep jobs, CI automation through Azure DevOps, and model deployment as a live inference endpoint.

The full MLOps workflow implemented:
```
code push → Azure DevOps pipeline → Azure ML training job → MLflow metrics → versioned model → deployed endpoint
```

---

## Repository Structure

```
├── src/
│   ├── train.py               # Training script
│   ├── score.py               # Scoring script for deployment
│   └── invoke_endpoint.py     # Endpoint invocation script
├── jobs/
│   ├── train_job.yml          # Azure ML command job definition
│   ├── sweep_job.yml          # Hyperparameter sweep job definition
│   └── deployment.yml         # Online deployment configuration
├── env/
│   ├── conda.yml              # Training environment
│   └── inference_conda.yml    # Inference environment
├── azure-pipelines.yml        # Azure DevOps CI pipeline
└── README.md
```

---

## Model Choice

**Model:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)  
**Solver:** `liblinear`  
**Why:** Fast convergence, interpretable, handles binary classification well, and doesn't require a GPU or 47GB of RAM to train. The assignment literally warned against SGDClassifier. Message received.

---

## Features Used

The merged feature datasets contain SBERT embeddings, TF-IDF vectors, sentiment scores, and length statistics. After a series of deeply humbling encounters with SIGKILL, the final feature configuration used for training is:

| Feature | Dimensions | Description |
|---|---|---|
| `sentiment_pos` | 1 | VADER positive sentiment score |
| `sentiment_neg` | 1 | VADER negative sentiment score |
| `sentiment_neu` | 1 | VADER neutral sentiment score |
| `sentiment_compound` | 1 | VADER compound sentiment score |
| `review_length_chars` | 1 | Character count of review |
| `review_length_words` | 1 | Word count of review |
| **Total** | **6** | |

### Why not SBERT or TF-IDF?

An attempt was made. Several attempts, actually. Each one was met with the same enthusiastic response from the OS: `SIGKILL. Possibly out of memory.` The available compute cluster does not have enough RAM to load SBERT vectors (384 floats × hundreds of thousands of rows) alongside TF-IDF matrices. The merged parquet files are large enough that even reading them without column filtering causes an OOM.

The sentiment and length features were chosen as the final configuration because they actually run, and they produce surprisingly solid results — which proves that sometimes less is more, and sometimes your cluster just has 7GB of RAM.

---

## Feature Experiments (Section IX.F)

Three feature configurations were attempted:

| Run | Features | Result |
|---|---|---|
| Run 1 | SBERT only | OOM — SIGKILL before first print statement |
| Run 2 | SBERT + TF-IDF | OOM — didn't even get to say hello |
| Run 3 | Sentiment + Length | ✅ Works. Ships. Deploys. Breathes. |

**Best feasible configuration:** Sentiment + Length features with LogisticRegression (liblinear). SBERT and TF-IDF were excluded due to memory constraints on the available compute cluster. The merged datasets do contain all engineered features — the limitation is at training time, not at feature engineering time.

---

## Hyperparameter Tuning

A sweep job was run over the following search space:

| Parameter | Type | Range |
|---|---|---|
| `C` | uniform | 0.1 → 10.0 |
| `max_iter` | choice | [100, 300, 500] |

**Sampling algorithm:** Random  
**Trials:** 6 total, 2 concurrent  
**Objective:** Maximize `val_accuracy`

### Sweep Results

| Run | C | max_iter | val_accuracy |
|---|---|---|---|
| **BEST** | **0.4765** | **300** | **0.8357** |
| 2 | 3.0648 | 500 | 0.8209 |
| 3 | 4.2413 | 100 | 0.8201 |
| 4 | 3.5194 | 500 | 0.8201 |
| 5 | 2.0434 | 100 | 0.8160 |
| 6 | 5.1961 | 100 | 0.8159 |

**Best hyperparameters:** `C=0.4765`, `max_iter=300`

---

## Final Model Performance

Trained using best hyperparameters from sweep, on 50,000 train / 15,000 val / 15,000 test rows.

| Split | Accuracy | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Train | 0.9015 | 0.8813 | 0.9168 | 0.9794 | 0.9470 |
| Validation | 0.8167 | 0.7145 | 0.8713 | 0.9216 | 0.8957 |
| Test | 0.8575 | 0.7817 | 0.8987 | 0.9437 | 0.9207 |
| **Deployment** | **0.8883** | — | **0.9962** | **0.8913** | **0.9408** |

**Training runtime:** ~16 seconds (yes, really)

The deployment split (most recent reviews by `review_year`) shows slightly different performance compared to the test set. This is consistent with the concept of **data drift** — language evolves, review behavior shifts, and a model trained on older data will naturally behave differently on newer data. This is not a bug. This is a Tuesday.

---

## Azure DevOps CI Pipeline

The training workflow is fully automated. Every push to the `assignment2_model_training` branch triggers the Azure DevOps pipeline which:

1. Installs the Azure ML CLI extension
2. Configures defaults for the workspace
3. Submits the training job via `az ml job create`
4. Streams logs in real time
5. Fails the pipeline if the Azure ML job doesn't complete successfully

**Pipeline runs:**
- `#20260407.1` — Initial pipeline setup ✅
- `#20260407.2` — Updated with best hyperparameters from sweep ✅

---

## Deployment

**Endpoint:** `amazon-review-endpoint-60304948`  
**Deployment:** `amazon-review-deployment`  
**Instance type:** `Standard_F2s_v2`  
**Auth mode:** Key

The model was registered as `amazon-review-sentiment-model` (version 1) and deployed to an Azure ML Managed Online Endpoint. The scoring script (`score.py`) uses the same feature construction logic as `train.py` — sentiment + length features only.

The endpoint was invoked using `invoke_endpoint.py` with the deployment dataset (`amazon_review_merged_features_deploy`), producing a deployment accuracy of **0.8883**.

The endpoint has been deleted after invocation as instructed. RIP little endpoint. You served your 8,000 predictions faithfully.

---

## Bonus Question — What Did We Do Wrong?

The one thing done "not correctly" in this assignment is **data leakage in the label creation step**.

In `create_binary_labels`, rating 3 (neutral) reviews are dropped:
```python
df = df[df["overall"].isin([1, 2, 4, 5])].copy()
```

This filtering is applied to train, val, test, AND the deployment split — but the decision to drop rating 3 was made by looking at the training data distribution. Applying this same filter to the test and deployment sets means the evaluation is not truly representative of all incoming data. In a real production system, the model would receive rating-3 reviews and would have to make a prediction, but it was never trained or evaluated on them.

The correct approach would be to either include rating 3 reviews in training (perhaps as a separate class or mapped to one of the binary classes), or to document explicitly that the model only handles non-neutral reviews and filter at inference time with a warning.

---

## Notes

- The `overall` column was missing from the initial merged datasets because `merge.py` didn't include it in the column selection from the length component. This was fixed and the Lab 4 pipeline was rerun for all 4 splits.
- The deployment split uses the most recent reviews by `review_year`, simulating data drift in production.
- Endpoint name required a suffix (`-60304948`) because Azure ML endpoint names must be unique across the entire Qatar Central region. Apparently someone else also wanted to name their endpoint `amazon-review-endpoint`. Bold choice.
