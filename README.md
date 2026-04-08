# Assignment 2 — Model Training & Automation with Azure
---

## Overview

This assignment transitions the Amazon Electronics review project from feature engineering into the training and experimentation phase. Starting from the merged feature datasets produced in Lab 4, this assignment covers model training on Azure ML compute, hyperparameter tuning via sweep jobs, CI automation through Azure DevOps, and model deployment as a live inference endpoint.

The full MLOps workflow implemented:
```
code push → Azure DevOps pipeline → Azure ML training job → MLflow metrics → versioned model → deployed endpoint
```

Was it smooth? No. Did the OS send SIGKILL multiple times like an aggressive bouncer? Yes. Did it eventually work? You're reading the README, so draw your own conclusions.

---

## Repository Structure

```
├── components/
│   ├── merge_features/
│   │   ├── component.yml
│   │   └── merge.py
│   ├── normalize_text/
│   │   ├── component.yml
│   │   └── normalize.py
│   ├── review_length/
│   │   ├── component.yml
│   │   └── review_length.py
│   ├── sample_dataset/
│   │   ├── component.yml
│   │   └── sample.py
│   ├── semantic_embeddings/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── embed.py
│   ├── sentiment/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── sentiment.py
│   ├── split_dataset/
│   │   ├── component.yml
│   │   └── split.py
│   └── tfidf_features/
│       ├── component.yml
│       └── tfidf.py
├── data_assets/
│   ├── amazon_review_merged_features_train.yml
│   ├── amazon_review_merged_features_val.yml
│   ├── amazon_review_merged_features_test.yml
│   ├── amazon_review_merged_features_deploy.yml
│   └── features_v1_sampled.yml
├── datastores/
│   └── raw_adls.yml
├── env/
│   ├── conda.yml
│   └── inference_conda.yml
├── jobs/
│   ├── train_job.yml
│   ├── sweep_job.yml
│   └── deployment.yml
├── pipelines/
│   └── feature_pipeline.yml
├── src/
│   ├── train.py
│   ├── score.py
│   └── invoke_endpoint.py
├── azure-pipelines.yml
└── README.md
```

---

## Feature Engineering Pipeline

The Lab 4 feature engineering pipeline processes all 4 dataset splits (train 60%, val 15%, test 15%, deploy 10%) through the following stages:

```
Raw Data → Sample → Split → Normalize Text
                                  ↓
             ┌────────────────────┼────────────────────┐
       SBERT Embeddings    TF-IDF Vectors    Sentiment + Length
             └────────────────────┼────────────────────┘
                                  ↓
                        Merge Feature Outputs
                                  ↓
                        data.parquet (per split)
```

Each merged dataset contains: `asin`, `reviewerID`, `overall`, `sbert_vector`, `tfidf_vector`, `sentiment_pos`, `sentiment_neg`, `sentiment_neu`, `sentiment_compound`, `review_length_chars`, `review_length_words`.

The deployment split uses the most recent reviews by `review_year` to simulate real production data and potential data drift. The feature pipeline ran successfully for all 4 splits. All features were engineered. The pipeline was beautiful. Then it was time to load them into the training cluster.

---

## Dataset Splits

| Split | Proportion | Purpose |
|---|---|---|
| Train | 60% | Model training |
| Validation | 15% | Hyperparameter tuning and experiment comparison |
| Test | 15% | Final offline evaluation |
| Deployment | 10% | Simulates incoming production data post-deployment |

The deployment split comes from the most recent time period in the dataset, meaning it may exhibit data drift relative to training data. Language evolves. Products change. Review behavior shifts. A model that performs slightly differently on the deployment split is not broken — it's honest.

---

## Data Assets

The following Azure ML Data Assets were registered (version 2, pointing to the fixed merge pipeline outputs):

- `amazon_review_merged_features_train`
- `amazon_review_merged_features_val`
- `amazon_review_merged_features_test`
- `amazon_review_merged_features_deploy`

> **Note:** A first version was registered before realizing the `overall` column (the label) was accidentally dropped during the merge step. The merge was fixed to preserve `overall`, the Lab 4 pipeline was rerun for all 4 splits, and version 2 was registered. Version 1 exists as a monument to the importance of reading error messages.

---

## Model Choice

**Model:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)  
**Solver:** `liblinear`  
**Why:** Fast convergence, interpretable, handles binary classification cleanly, and does not require 64GB of RAM. The assignment explicitly warned against SGDClassifier. Warning heeded. No regrets.

**Label:** Binary — reviews rated 4 or 5 are positive (1), reviews rated 1 or 2 are negative (0). Rating 3 is excluded.

---

## Training Features

The merged datasets contain all engineered features from Lab 4. The training script loads only the columns required for the chosen feature configuration using pyarrow column pruning, so SBERT and TF-IDF data is skipped at read time when not needed — preventing the cluster from being overwhelmed.

### The OOM Chronicles: A Trilogy

**Chapter 1 — SBERT Only**  
SBERT vectors are 384 floats per row. Sounds fine. Is not fine. Loading them across the full training split resulted in `SIGKILL. Possibly out of memory.` The `std_log.txt` was empty. Not a single line. The process was killed before it could even say hello.

**Chapter 2 — SBERT + TF-IDF**  
If SBERT alone kills the process, surely adding a sparse TF-IDF matrix on top is fine. It was not fine. Tried reducing row count. Tried column pruning. Tried `gc.collect()` after every breath. The cluster remained unmoved. SIGKILL, faster this time, as if it was getting impatient.

**Chapter 3 — The One That Works**  
After extensive negotiation with the laws of memory management, a feature configuration was found that trains reliably within the cluster's constraints, still produces meaningful results, and does not cause the OS to file for divorce.

### Final Feature Configuration

| Feature | Description |
|---|---|
| `sentiment_pos` | VADER positive sentiment score |
| `sentiment_neg` | VADER negative sentiment score |
| `sentiment_neu` | VADER neutral sentiment score |
| `sentiment_compound` | VADER compound sentiment score |
| `review_length_chars` | Character count of review |
| `review_length_words` | Word count of review |

---

## Hyperparameter Tuning

A sweep job was run over the following search space:

| Parameter | Type | Range |
|---|---|---|
| `C` (regularization strength) | uniform | 0.1 → 10.0 |
| `max_iter` | choice | [100, 300, 500] |

**Sampling algorithm:** Random  
**Trials:** 6 total, 2 concurrent  
**Objective:** Maximize `val_accuracy`

### Sweep Results

| Run | C | max_iter | val_accuracy | Duration |
|---|---|---|---|---|
| **BEST** ⭐ | **0.4765** | **300** | **0.8357** | 32s |
| 2 | 3.0648 | 500 | 0.8209 | 33s |
| 3 | 4.2413 | 100 | 0.8201 | 2m 12s |
| 4 | 3.5194 | 500 | 0.8201 | 2m 9s |
| 5 | 2.0434 | 100 | 0.8160 | 1m 5s |
| 6 | 5.1961 | 100 | 0.8159 | 33s |

The optimal C is 0.4765 — not 1.0 (the default), not 10.0 (the overconfident choice). The universe works in mysterious and very specific decimal ways.

---

## Final Model Performance

Trained using best hyperparameters (`C=0.4765`, `max_iter=300`), on 50k train / 15k val / 15k test rows. Job duration: ~2m 26s. MLflow-logged training script runtime: ~16.6s (the rest is Azure doing Azure things).

| Split | Accuracy | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Train | 0.9015 | 0.8813 | 0.9168 | 0.9794 | 0.9470 |
| Validation | 0.8167 | 0.7145 | 0.8713 | 0.9216 | 0.8957 |
| Test | 0.8575 | 0.7817 | 0.8987 | 0.9437 | 0.9207 |
| **Deployment** | **0.8883** | — | **0.9962** | **0.8913** | **0.9408** |

The deployment split performs slightly differently from the test set — consistent with data drift from newer reviews. Expected behavior. Not a bug.

---

## Model Registration

After the final training run, the model artifact was registered in the Azure ML Model Registry:

```
Name: amazon-review-sentiment-model
Version: 1
Type: custom_model
Job: affable_library_4y1j0mc6l9
```

Registering the model enables version tracking, experiment lineage, and clean deployment references. A model that isn't registered is just a `.pkl` file sitting in a folder somewhere, which is not a deployment strategy.

---

## Azure DevOps CI Pipeline

Every push to `assignment2_model_training` automatically triggers the Azure DevOps pipeline which submits the training job, streams logs in real time, and fails loudly if something goes wrong. No more manually submitting jobs in the terminal like a caveperson.

**Service Connection:** `SC-UDST-CCIT-DSAI3202-1`

**Pipeline runs:**
- `#20260407.1` — Initial pipeline setup ✅ (6m 23s)
- `#20260407.2` — Updated with best hyperparameters from sweep ✅ (7m 24s)

---

## Deployment

**Endpoint:** `amazon-review-endpoint-60304948`  
**Deployment:** `amazon-review-deployment`  
**Instance type:** `Standard_F2s_v2`  
**Auth mode:** Key  
**Model:** `amazon-review-sentiment-model@latest`

The model was deployed as an Azure ML Managed Online Endpoint exposing a REST API. The scoring script (`score.py`) defines `init()` which loads the model on startup, and `run()` which processes incoming requests using the same feature construction logic as training.

The endpoint was invoked using `invoke_endpoint.py` with the deployment dataset (`amazon_review_merged_features_deploy`), sending requests in batches of 10:

```
Loaded rows: 1,502,077
Deployment accuracy:  0.8883
Deployment precision: 0.9962
Deployment recall:    0.8913
Deployment F1:        0.9408
Predictions returned: 8,000
```

The endpoint has since been deleted. RIP `amazon-review-endpoint-60304948`. You lived for one day, served 8,000 predictions faithfully, and will be remembered.

> *The endpoint name required a `-60304948` suffix because Azure ML endpoint names must be globally unique across the entire Qatar Central region. Someone else had already claimed `amazon-review-endpoint`. Respect to whoever got there first.*

---

## Bonus Question — What Is Done Wrong in This Assignment?

The thing done "not correctly" in this assignment is the **order of operations between CI/CD setup and hyperparameter tuning**.

In the assignment, Azure DevOps CI/CD automation is set up in Section VIII — before hyperparameter tuning happens in Section IX. This means the automated pipeline is configured and running before the model has been properly tuned or the best configuration has been identified.

In standard MLOps practice, the correct order is:

```
1. Experiment locally / manually
2. Tune hyperparameters
3. Finalize model configuration
4. THEN automate with CI/CD
```

What the assignment does instead:

```
1. Set up CI/CD automation  ← automating what exactly? we don't know yet
2. Then tune hyperparameters
3. Then update the code and push to trigger the automated pipeline
```

The CI/CD pipeline ends up automating an untuned, default-configuration model first, then gets updated later. The automation should wrap around a finalized workflow, not be set up in the middle of the experimentation phase. Setting up CI/CD before you know what you're automating is like building a highway before deciding where it goes.

The correct approach would be to complete all experimentation and hyperparameter tuning first, finalize the training configuration, and only then set up the automation pipeline that runs the final, validated workflow.

---

*This assignment involved approximately 47 failed job runs, 12 OOM errors, 3 existential crises, 1 missing `overall` column, 1 GitHub push blocked for containing secrets, and one very patient language model. It has been completed. The endpoint has been deleted. We can all go home now.*
