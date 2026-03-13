import argparse
import os
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--results_out", type=str, required=True)
    parser.add_argument("--features_out", type=str, required=True)
    parser.add_argument("--population_size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    id_col = "unit_number"
    target_col = "RUL"

    ids = df[[id_col]].copy()
    y = df[target_col].copy()
    X = df.drop(columns=[id_col, target_col]).copy()

    n_features = X.shape[1]
    if n_features == 0:
        raise ValueError("No features available for modeling.")

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected_idx) == 0:
            return (-9999,)

        X_selected = X.iloc[:, selected_idx]

        model = RandomForestRegressor(
            n_estimators=80,
            random_state=42,
            n_jobs=-1
        )

        scores = cross_val_score(
            model,
            X_selected,
            y,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        score = scores.mean()
        penalty = len(selected_idx) / n_features
        fitness = score - 0.05 * penalty
        return (fitness,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=args.population_size)
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=args.generations,
        verbose=True
    )

    best_individual = tools.selBest(population, k=1)[0]
    selected_cols = X.columns[[i for i, bit in enumerate(best_individual) if bit == 1]]

    if len(selected_cols) == 0:
        selected_cols = X.columns[:min(5, len(X.columns))]

    X_final = X[selected_cols].copy()
    final_features_df = pd.concat([ids, X_final, y], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": model_name,
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "MAE": float(mean_absolute_error(y_test, preds)),
            "R2": float(r2_score(y_test, preds)),
            "Selected_Features": int(X_final.shape[1])
        })

    results_df = pd.DataFrame(results)

    os.makedirs(args.results_out, exist_ok=True)
    os.makedirs(args.features_out, exist_ok=True)

    results_df.to_parquet(os.path.join(args.results_out, "data.parquet"), index=False)
    final_features_df.to_parquet(os.path.join(args.features_out, "data.parquet"), index=False)

    print(results_df)
    print("Final selected feature count:", X_final.shape[1])


if __name__ == "__main__":
    main()