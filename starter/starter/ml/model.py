"""
ML model training, inference, and metrics calculation.

Provides functions to train a RandomForestClassifier model,
make predictions, and compute performance metrics.

// Renamed the import to name the functions more specific
"""
from sklearn.metrics import (
    fbeta_score as calculate_fbeta,
    precision_score as calculate_precision,
    recall_score as calculate_recall,
)
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    # Create model
    model = RandomForestClassifier()

    # Train model
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta_score = calculate_fbeta(y, preds, beta=1, zero_division=1)
    precision_score = calculate_precision(y, preds, zero_division=1)
    recall_score = calculate_recall(y, preds, zero_division=1)

    return precision_score, recall_score, fbeta_score


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    # Make predictions
    preds = model.predict(X)

    return preds


if __name__ == "__main__":
    import numpy as np

    # Create dummy test data
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[2, 3], [6, 7]])
    y_test = np.array([0, 1])

    # Train model
    model = train_model(X_train, y_train)
    print("Model trained successfully!")

    # Make predictions
    preds = inference(model, X_test)
    print(f"Predictions: {preds}")

    # Calculate metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-Beta: {fbeta:.3f}")
