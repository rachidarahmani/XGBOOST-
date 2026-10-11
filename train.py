"""Entraîne le modèle XGBoost (pipeline du notebook) et sauvegarde l'artefact."""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RS = 42
DATA_DIR = Path("data")


def load_data():
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red["type"], white["type"] = 0, 1  # rouge = 0, blanc = 1
    df = pd.concat([red, white], ignore_index=True)
    df.columns = df.columns.str.replace(" ", "_").str.lower()  # noms compatibles avec l'API (pH -> ph)

    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # suppression des doublons
    print(f"Doublons supprimés : {n - len(df)} -> {len(df)} vins")

    df["good"] = (df["quality"] >= 7).astype(int)
    return df


def iqr_bounds(s, k=3):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def main():
    df = load_data()
    features_all = [c for c in df.columns if c not in ("quality", "good")]
    num_cols = [c for c in features_all if c != "type"]

    X, y = df[features_all], df["good"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RS
    )

    # Écrêtage 3xIQR, bornes calculées sur le TRAIN uniquement
    bounds = {c: iqr_bounds(X_train[c], 3) for c in num_cols}
    X_train_c, X_test_c = X_train.copy(), X_test.copy()
    for c, (lo, hi) in bounds.items():
        X_train_c[c] = X_train[c].clip(lo, hi)
        X_test_c[c] = X_test[c].clip(lo, hi)

    # Optimisation des hyperparamètres (RandomizedSearchCV, score F1)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=1),
        {
            "n_estimators": [200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "scale_pos_weight": [1, pos_weight],
        },
        n_iter=15, cv=skf, scoring="f1", random_state=RS, n_jobs=1,
    )
    search.fit(X_train_c, y_train)
    model = search.best_estimator_
    print("Meilleurs paramètres :", search.best_params_)
    print("F1 CV :", round(search.best_score_, 4))

    # Évaluation unique sur le test
    pred = model.predict(X_test_c)
    proba = model.predict_proba(X_test_c)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    print("Test :", metrics)

    importances = dict(zip(features_all, map(float, model.feature_importances_)))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features_all,
            "lower": {c: float(b[0]) for c, b in bounds.items()},
            "upper": {c: float(b[1]) for c, b in bounds.items()},
            "metrics": metrics,
            "importances": importances,
        },
        "models/wine_model.joblib",
    )
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Modèle sauvegardé dans models/wine_model.joblib")


if __name__ == "__main__":
    main()
