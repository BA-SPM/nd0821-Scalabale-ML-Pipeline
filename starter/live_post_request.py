"""
Send one POST request to the deployed census income API.
"""

import requests


API_URL_ROOT = "https://nd0821-scalabale-ml-pipeline.onrender.com/"  # Root URL for the API
API_URL_PREDICT = f"{API_URL_ROOT}predict"


def build_payload() -> dict:
    """Create one example request payload for the live API."""

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


def main() -> None:
    """
    Send the request and print status code and response body.
    """

    payload = build_payload()
    response = requests.post(API_URL_PREDICT, json=payload, timeout=30)

    print(f"Status code: {response.status_code}")
    print("Response body:")
    print(response.text)


if __name__ == "__main__":
    main()
