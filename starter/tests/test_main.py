"""Unit tests for FastAPI endpoints in main.py."""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient

try:
    import starter.main as main
except ModuleNotFoundError:
    # Fallback for runs where pytest rootdir is the starter/ folder.
    project_dir = Path(__file__).resolve().parents[1]
    if str(project_dir) not in sys.path:
        sys.path.append(str(project_dir))
    import main as main


def _valid_payload() -> dict:
    """
    Create one valid API request payload.
    """

    return {
        "age": 37,
        "workclass": "Private",
        "final_weight": 34146,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }


def _mock_prediction_dependencies(
    monkeypatch,
    model_prediction: int,
    decoded_label: str,
) -> None:
    """
    Replace model-related dependencies in main.py for deterministic API tests.
    """

    fake_lb = Mock()
    fake_lb.inverse_transform.return_value = np.array([decoded_label])

    monkeypatch.setattr(main, "MODEL", object())
    monkeypatch.setattr(main, "ENCODER", object())
    monkeypatch.setattr(main, "LABEL_BINARIZER", fake_lb)
    monkeypatch.setattr(
        main,
        "process_data",
        lambda *args, **kwargs: (np.array([[1, 2]]), None, None, None),
    )
    monkeypatch.setattr(
        main,
        "inference",
        lambda model, data: np.array([model_prediction]),
    )


def test_get_root_returns_help_text():
    """
    Test that GET / returns the plain-text help message.
    """

    client = TestClient(main.app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Welcome to the Census Income Prediction API" in response.text


def test_post_predict_returns_less_equal_50k(monkeypatch):
    """
    Test POST /predict for <=50K prediction output.
    """

    _mock_prediction_dependencies(
        monkeypatch=monkeypatch,
        model_prediction=0,
        decoded_label="<=50K",
    )

    client = TestClient(main.app)
    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_post_predict_returns_greater_50k(monkeypatch):
    """
    Test POST /predict for >50K prediction output.
    """

    _mock_prediction_dependencies(
        monkeypatch=monkeypatch,
        model_prediction=1,
        decoded_label=">50K",
    )

    client = TestClient(main.app)
    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
