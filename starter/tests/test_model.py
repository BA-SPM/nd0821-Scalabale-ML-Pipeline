"""Unit tests for model training, inference, and metrics computation."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Import path setup for test execution
project_dir = Path(__file__).resolve().parents[1]
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

from starter.ml.model import train_model, inference, compute_model_metrics


class TestTrainModel:
    """Tests for train_model function."""

    def test_train_model_returns_model_object(self):
        """Test that train_model returns a trained model object."""
        # Arrange: Create dummy training data
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])

        # Act: Train the model
        model = train_model(X_train, y_train)

        # Assert: Model is not None and has predict method
        assert model is not None
        assert hasattr(model, 'predict')
        assert callable(getattr(model, 'predict'))

    def test_train_model_output_type(self):
        """Test that train_model output is a RandomForestClassifier."""
        # Arrange
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])

        # Act
        model = train_model(X_train, y_train)

        # Assert
        assert type(model).__name__ == 'RandomForestClassifier'


class TestInference:
    """Tests for inference function."""

    def test_inference_returns_predictions(self):
        """Test that inference returns predictions array."""
        # Arrange: Create and train model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model = train_model(X_train, y_train)

        X_test = np.array([[2, 3], [6, 7]])

        # Act: Make predictions
        preds = inference(model, X_test)

        # Assert: Predictions exist and have correct shape
        assert preds is not None
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X_test)

    def test_inference_output_shape_matches_input(self):
        """Test that inference output has same length as input."""
        # Arrange
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        model = train_model(X_train, y_train)

        X_test = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])

        # Act
        preds = inference(model, X_test)

        # Assert
        assert preds.shape[0] == X_test.shape[0]


class TestComputeMetrics:
    """Tests for compute_model_metrics function."""

    def test_compute_model_metrics_returns_three_values(self):
        """Test that compute_model_metrics returns precision, recall, fbeta."""
        # Arrange: Perfect predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        # Act: Compute metrics
        precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

        # Assert: All three metrics are floats
        assert isinstance(precision, (float, np.floating))
        assert isinstance(recall, (float, np.floating))
        assert isinstance(fbeta, (float, np.floating))

    def test_compute_model_metrics_perfect_prediction(self):
        """Test that perfect predictions yield 1.0 for all metrics."""
        # Arrange: Perfect predictions
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1, 0])

        # Act
        precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

        # Assert: All metrics should be 1.0
        assert precision == 1.0
        assert recall == 1.0
        assert fbeta == 1.0

    def test_compute_model_metrics_metrics_in_valid_range(self):
        """Test that metrics are between 0 and 1."""
        # Arrange: Some predictions correct, some wrong
        y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])

        # Act
        precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

        # Assert: All metrics in valid range [0, 1]
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= fbeta <= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
