# Lab 5 — Scalable Feature Extraction and Selection for Predictive Maintenance


> **Objective:** Build a scalable, reproducible feature engineering pipeline on the NASA C-MAPSS Turbofan dataset to predict **Remaining Useful Life (RUL)** of aircraft engines — fast, modular, and production-ready on Azure.

---

## Overview

Industrial sensors generate high-dimensional time-series data. Raw signals alone cannot be fed into ML models — they must be transformed into meaningful features. This lab implements an end-to-end pipeline that:

1. Ingests the NASA C-MAPSS FD001 dataset from Azure Data Lake Storage
2. Preprocesses the multi-sensor time-series data
3. Extracts time-series features with **tsfresh**
4. Reduces dimensionality using filter-based methods (Variance Threshold → Correlation Filtering → Mutual Information)
5. Evolves an optimal feature subset using a **Genetic Algorithm (DEAP)**
6. Trains and evaluates multiple regression models (Random Forest, Gradient Boosting, XGBoost)
7. Saves the final feature set and model results back to the Azure Gold layer

---

## Repository Structure

```
.
├── components/
│   ├── helpfulness_ratio/          # Engine Trend Features
│   │   ├── helpfulness.py
│   │   └── component.yml
│   ├── merge_features/             # Merge All Feature Tables
│   │   ├── merge_features.py
│   │   └── component.yml
│   ├── normalize_text/             # Create RUL Labels
│   │   ├── normalize.py
│   │   └── component.yml
│   ├── review_length/              # Handcrafted Engine Summary Features
│   │   ├── review_length.py
│   │   └── component.yml
│   ├── semantic_embeddings/        # Extract TSFresh Features
│   │   ├── semantic_embeddings.py
│   │   ├── conda.yml
│   │   └── component.yml
│   ├── sentiment/                  # DEAP Genetic Algorithm + Model Training
│   │   ├── sentiment.py
│   │   ├── conda.yml
│   │   └── component.yml
│   ├── split_dataset/              # Split Turbofan Dataset
│   │   ├── split.py
│   │   └── component.yml
│   └── tfidf_features/             # Filter Features (Variance / MI)
│       ├── tfidf.py
│       └── component.yml
├── data/
│   ├── rul_fd001.yml
│   ├── test_fd001.yml
│   ├── train_fd001.yml
│   └── turbofan_features_v1.yml
├── datastores/
│   ├── curated_adls.yml
│   └── raw_adls.yml
├── environments/
│   ├── deap-xgb-env.yml            # Conda env: DEAP + XGBoost pipeline
│   └── tsfresh-env.yml             # Conda env: tsfresh extraction
├── notebooks/
│   └── lab5_predictive_maintainance.ipynb
├── pipelines/
│   └── pipeline.yml
├── .gitignore
└── README.md
```

---

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001 subset)**

| Property | Value |
|---|---|
| Source | [NASA PCoE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| Engines (train) | 100 |
| Features | 3 operating settings + 21 sensor measurements |
| Target | Remaining Useful Life (RUL) in cycles |
| Format | Space-separated `.txt` → Parquet (Bronze → Silver → Gold) |

The raw files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) are ingested from the Azure Data Lake `raw` container and processed through Bronze → Silver → Gold layers.

---

## Pipeline Architecture

```
Raw ADLS (Bronze)
       │
       ▼
  split_dataset          ← Loads + renames columns, adds RUL labels
       │
       ▼
  normalize_text         ← Computes RUL = max_cycle − current_cycle
       │
  ┌────┴──────────────────────────┐
  ▼                               ▼
semantic_embeddings         review_length + helpfulness_ratio
(tsfresh MinimalFC)         (handcrafted summary & trend features)
  │                               │
  └────────────┬──────────────────┘
               ▼
         merge_features           ← Combines all feature tables
               │
               ▼
         tfidf_features           ← Variance Threshold → Correlation → Mutual Information
               │
               ▼
           sentiment              ← DEAP Genetic Algorithm + Model Training
               │
               ▼
       Gold Layer (ADLS)          ← Final feature set + model results saved as Parquet
```

---

## Components

### `split_dataset` — Split Turbofan Dataset
Loads the raw FD001 text files, renames columns (`unit_number`, `time_in_cycles`, `op_setting_1–3`, `sensor_1–21`), and splits into train/test/RUL partitions. Saves to Silver layer as Parquet.

**Inputs:** `train_data`, `test_data`, `rul_data`  
**Parameters:** `sample_engines` (default 40), `seed` (default 42)

---

### `normalize_text` — Create RUL Labels
Computes the Remaining Useful Life for each engine in the training set:  
`RUL = max_cycle_for_engine − current_time_in_cycles`

**Inputs:** `train_data`, `test_data`, `rul_data`

---

### `semantic_embeddings` — Extract TSFresh Features
Runs **tsfresh** `MinimalFCParameters` on all 24 sensor/setting columns per engine to generate a compact, statistically-derived feature matrix. Uses `impute()` to handle any NaN values.

**Environment:** `tsfresh-env` (Python 3.8, tsfresh, scikit-learn, pandas)  
**Output shape:** `(100 engines × ~120 features)`

---

### `review_length` — Handcrafted Engine Summary Features
Computes per-engine summary statistics manually: engine life cycles, sensor means, standard deviations, and delta (last − first) for key sensors (11, 12, 15).

---

### `helpfulness_ratio` — Engine Trend Features
Extracts first-value, last-value, and delta for sensors 11, 12, and 15 — capturing the degradation trend over each engine's lifetime.

---

### `tfidf_features` — Filter Features
Applies a 3-stage filter pipeline to reduce dimensionality before the Genetic Algorithm:

| Step | Method | Result |
|---|---|---|
| 1 | Variance Threshold (`threshold=0.0`) | Removes zero-variance features |
| 2 | Correlation Filter (`threshold > 0.95`) | Removes redundant correlated features |
| 3 | Mutual Information (`top_k=50`) | Keeps the 50 most informative features |

---

### `merge_features` — Merge All Feature Tables
Concatenates the tsfresh features, handcrafted summary features, and trend features into a single unified feature table (deduplicating the `RUL` target column).

---

### `sentiment` — DEAP Genetic Algorithm + Model Training
Applies a binary Genetic Algorithm to evolve the optimal feature subset, then trains and evaluates multiple models.

**GA Configuration:**

| Parameter | Value |
|---|---|
| Population size | 12 (default) |
| Generations | 6 (default) |
| Crossover probability | 0.5 |
| Mutation probability | 0.2 (bit-flip, `indpb=0.05`) |
| Selection | Tournament (`tournsize=3`) |
| Fitness | neg-RMSE − 0.05 × (feature count ratio) |

**Models trained:** Random Forest, Gradient Boosting, XGBoost

---

## Environments

### `tsfresh-env.yml`
Used by the `semantic_embeddings` component for feature extraction.
```yaml
name: tsfresh-env
dependencies:
  - python=3.8
  - pip:
    - pandas
    - numpy
    - pyarrow
    - tsfresh
    - scikit-learn
```

### `deap-xgb-env.yml`
Used by the `sentiment` component for GA optimization and model training.
```yaml
name: deap-xgb-env
dependencies:
  - python=3.8
  - pip:
    - pandas
    - numpy
    - pyarrow
    - scikit-learn
    - deap
    - xgboost
```

---

## Results

The best model was **Gradient Boosting**, achieving near-perfect RUL prediction on the FD001 test set.

| Model | RMSE | MAE | R² | Selected Features |
|---|---|---|---|---|
| Gradient Boosting | ~3.0 | ~1.86 | **0.995** | 20 |
| Random Forest | — | — | — | 20 |
| XGBoost | — | — | — | 20 |

The Gradient Boosting model explains **99.5% of variance** in RUL — indicating the feature engineering pipeline produces a highly informative and compact representation of engine degradation.

**Pipeline runtime** is tracked end-to-end using `time.time()` and printed at the end of the notebook.

---

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CloudComputing_Labs-Assignment1.git
cd CloudComputing_Labs-Assignment1
```

### 2. Download the dataset

Download the **C-MAPSS FD001** subset from the [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (item #6).

Upload the three files to your Azure Data Lake `raw` container:
```
raw/lab5_turbofan/train_FD001.txt
raw/lab5_turbofan/test_FD001.txt
raw/lab5_turbofan/RUL_FD001.txt
```

### 3. Configure your storage credentials

In the notebook (`notebooks/lab5_predictive_maintainance.ipynb`), update:
```python
storage_account_name = "<your-storage-account>"
storage_account_key  = "<your-storage-key>"
```

Or register the datastores using the provided YAML files:
```bash
az ml datastore create --file datastores/raw_adls.yml
az ml datastore create --file datastores/curated_adls.yml
```

### 4. Run the notebook

Open `notebooks/lab5_predictive_maintainance.ipynb` in Azure Databricks or Azure ML and run all cells in order.

### 5. Run the full Azure ML pipeline (optional)

```bash
az ml job create --file pipelines/pipeline.yml
```

---

## Azure Setup

| Resource | Purpose |
|---|---|
| Azure Data Lake Storage Gen2 | Raw (Bronze), Processed (Silver), Curated (Gold) layers |
| Azure Databricks / Azure ML | Spark-based preprocessing and notebook execution |
| Azure ML Pipelines | Orchestrate modular components end-to-end |
| Azure ML Environments | Reproducible conda environments per component |

Data flows through three ADLS containers:
- `raw` → original FD001 `.txt` files
- `processed` → cleaned Parquet, RUL-labeled data
- `curated` → final feature set and model results

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyspark` | Distributed data loading and preprocessing |
| `tsfresh` | Automated time-series feature extraction |
| `deap` | Genetic Algorithm for feature selection |
| `scikit-learn` | Filter methods, model training, evaluation |
| `xgboost` | XGBoost regressor |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` | Visualization |
| `pyarrow` | Parquet I/O |
