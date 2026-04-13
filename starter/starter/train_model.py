"""
Train model on clean census data and save model artifacts.
"""


# ============================
# Imports
# ============================
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Import path setup for direct script execution
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from starter.ml.data import process_data
from starter.ml.model import train_model


# ============================
# Path setup and constants
# ============================
PROJECT_DIR = Path(__file__).resolve().parents[1]
# TODO: Make data reading and model saving paths more flexible, e.g.,
# via command-line arguments or config file.
DATA_PATH = PROJECT_DIR / "data" / "census.csv"
MODEL_DIR = PROJECT_DIR / "model"

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# ============================
# Functions
# ============================
def save_pickle_file(obj, file_path):
    """
    Save one Python object as a pickle file.
    """

    # Binary artifact writing
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)

    print(f"Saved artifact to {file_path}")


def load_pickle_file(file_path):
    """
    Load one Python object from a pickle file.
    """

    # Binary artifact reading
    with open(file_path, "rb") as file:
        obj = pickle.load(file)

    print(f"Loaded artifact from {file_path}")

    return obj


def load_clean_data(data_path=DATA_PATH):
    """
    Load clean census data from CSV file.
    """

    data = pd.read_csv(data_path)
    print(f"Loaded clean data from {data_path}")

    return data


def split_train_test(data):
    """Split input data into train and test sets."""

    return train_test_split(data, test_size=0.20, random_state=42)


def process_training_data(train_df):
    """
    Process training data and return features, labels, encoder, and label_binarizer.
    """
    # Training preprocessing
    return process_data(
        train_df,
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=True,
    )


def process_test_data(test_df, encoder, label_binarizer):
    """
    Process test data using fitted encoder and label binarizer.
    """
    # Test preprocessing
    return process_data(
        test_df,
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=label_binarizer,
    )


def save_artifacts(model, encoder, label_binarizer, model_dir=MODEL_DIR):
    """
    Save model artifacts into model directory.
    """

    # Ensure model directory exists
    model_dir.mkdir(exist_ok=True)

    # Artifact mapping
    artifacts = {
        "model.pkl": model,
        "encoder.pkl": encoder,
        "lb.pkl": label_binarizer,
    }
    for file_name, artifact in artifacts.items():
        save_pickle_file(artifact, model_dir / file_name)

    print(f"All artifacts saved to {model_dir}")


def load_artifacts(model_dir=MODEL_DIR):
    """
    Load model artifacts from model directory.
    """

    # Artifact loading
    model = load_pickle_file(model_dir / "model.pkl")
    encoder = load_pickle_file(model_dir / "encoder.pkl")
    label_binarizer = load_pickle_file(model_dir / "lb.pkl")

    print(f"All artifacts loaded from {model_dir}")

    return model, encoder, label_binarizer


def run_training_pipeline():
    """
    Run full training pipeline and save all artifacts.
    """
    # Data loading
    data = load_clean_data()

    # Data split
    train_df, test_df = split_train_test(data)

    # Training preprocessing
    X_train, y_train, encoder, label_binarizer = process_training_data(train_df)

    # Test preprocessing
    process_test_data(test_df, encoder, label_binarizer)

    # Model fitting
    model = train_model(X_train, y_train)

    # Artifact persistence
    save_artifacts(model, encoder, label_binarizer)

    print("Training complete. Artifacts saved to 'model/' directory.")


def main():
    """
    Script entry point for local training run.
    """
    # Pipeline execution
    run_training_pipeline()


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    main()
