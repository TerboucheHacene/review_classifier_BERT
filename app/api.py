from fastapi import FastAPI, Request


from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Optional, List
import json

from review_cl.inference import model_fn, predict_fn
from app.schemas import PredictReview

# Define application
app = FastAPI(
    title="Review Sentiment Classification",
    description="Predict sentiment of a review text",
    version="0.1",
)


@app.on_event("startup")
def load_artifacts():
    global artifacts
    artifacts = json.loads(open("logs/config.json", "r").read())
    artifacts["model"] = model_fn(model_dir="logs/")
    print("Ready for inference!")


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request):
    """Health check."""
    print(request.method)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.get("/params/{param}", tags=["Parameters"])
@construct_response
def _param(request: Request, param: str):
    """Get a specific parameter's value used for a run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {param: artifacts.get(param, "")},
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictReview) -> Dict:
    """Predict tags for a list of texts using the best run."""
    # Predict
    reviews = [item.review for item in payload.reviews]
    predictions = predict_fn(input_data=reviews, model=artifacts["model"])
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": predictions,
    }
    return response
