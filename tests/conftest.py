import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol",
]
ALL_COLS = FEATURES + ["type"]


@pytest.fixture(scope="session")
def client(tmp_path_factory, monkeypatch_session):
    """API testée avec un petit modèle factice (pas besoin des données réelles)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, len(ALL_COLS))), columns=ALL_COLS)
    y = (X["alcohol"] > 0.5).astype(int)
    model = XGBClassifier(n_estimators=10, random_state=0).fit(X, y)

    path = tmp_path_factory.mktemp("model") / "wine_model.joblib"
    joblib.dump(
        {
            "model": model,
            "features": ALL_COLS,
            "lower": {c: 0.0 for c in FEATURES},
            "upper": {c: 1000.0 for c in FEATURES},
            "metrics": {"f1": 0.6},
            "importances": {c: 0.1 for c in ALL_COLS},
        },
        path,
    )
    monkeypatch_session.setenv("MODEL_PATH", str(path))

    from app.main import app, get_artifact

    get_artifact.cache_clear()
    return TestClient(app)


@pytest.fixture(scope="session")
def monkeypatch_session():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture
def valid_payload():
    return {
        "fixed_acidity": 7.0, "volatile_acidity": 0.3, "citric_acid": 0.3,
        "residual_sugar": 2.0, "chlorides": 0.05, "free_sulfur_dioxide": 30.0,
        "total_sulfur_dioxide": 100.0, "density": 0.995, "ph": 3.2,
        "sulphates": 0.5, "alcohol": 11.0, "wine_type": "white",
    }
