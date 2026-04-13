"""
FastAPI app for census income model inference.
This app loads the trained model and encoders, processes incoming data, and returns predictions."""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import CATEGORICAL_FEATURES, load_artifacts


class CensusRecord(BaseModel):
    """
    Input data for one census income prediction.
    """

    age: int
    workclass: str
    final_weight: int = Field(alias="fnlgt")
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )


app = FastAPI(title="Census Income Prediction API")


try:
    MODEL, ENCODER, LABEL_BINARIZER = load_artifacts()
except FileNotFoundError:
    MODEL, ENCODER, LABEL_BINARIZER = None, None, None


@app.get("/", response_class=PlainTextResponse)
def welcome_root() -> str:
    """
    Return a simple usage help message for root.
    """

    return (
        "Welcome to the Census Income Prediction API\n"
        "===========================================\n\n"
        "How to use:\n"
        "1) GET \"/\"           -> show this help\n"
        "2) POST \"/predict\"   -> run one inference\n"

        "Terminal example:\n"
        "curl \"http://127.0.0.1:8000/predict\" `\n"
        "  -H \"Content-Type: application/json\" `\n"
        "  -d '{\n"
        "    \"age\": 37,\n"
        "    \"workclass\": \"Private\",\n"
        "    \"final_weight\": 34146,\n"
        "    \"education\": \"Bachelors\",\n"
        "    \"education-num\": 13,\n"
        "    \"marital-status\": \"Married-civ-spouse\",\n"
        "    \"occupation\": \"Exec-managerial\",\n"
        "    \"relationship\": \"Husband\",\n"
        "    \"race\": \"White\",\n"
        "    \"sex\": \"Male\",\n"
        "    \"capital-gain\": 0,\n"
        "    \"capital-loss\": 0,\n"
        "    \"hours-per-week\": 40,\n"
        "    \"native-country\": \"United-States\"\n"
        "  }'\n\n"
    )


@app.post("/predict")
def predict_salary(record: CensusRecord) -> dict[str, str]:
    """
    Run model inference for one census record.
    """

    if MODEL is None or ENCODER is None or LABEL_BINARIZER is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not available.")

    input_data = pd.DataFrame([record.model_dump(by_alias=True)])
    processed_data, _, _, _ = process_data(
        input_data,
        categorical_features=CATEGORICAL_FEATURES,
        training=False,
        encoder=ENCODER,
        lb=LABEL_BINARIZER,
    )

    prediction = inference(MODEL, processed_data)[0]
    prediction_label = LABEL_BINARIZER.inverse_transform(np.array([prediction]))[0]

    return {"prediction": prediction_label}


def main() -> None:
    """Run the FastAPI app locally with uvicorn."""

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
