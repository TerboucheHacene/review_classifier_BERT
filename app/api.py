from fastapi import FastAPI, Request, Depends, status, Path
from fastapi.exceptions import HTTPException
from typing import Dict, Optional
from transformers import RobertaForSequenceClassification
from review_cl.inference import model_fn, predict_fn
from app.schemas import PredictReviewInput, PredictReviewOutput, Settings

# Define application
app = FastAPI(
    title="Review Sentiment Classification",
    description="Predict sentiment of a review text",
    version="0.1",
)

settings = Settings()


class ReviewClassification:
    model: Optional[RobertaForSequenceClassification]
    artifacts: Optional[Dict]

    def load_model(self):
        model_and_artifacts = model_fn(
            model_dir=settings.model_path,
            model_config_path=settings.config_file,
            model_name=settings.MODEL_NAME,
        )
        model, artifacts = model_and_artifacts
        self.model = model.eval()
        self.artifacts = artifacts
        self.properties = list(artifacts.keys())

    async def predict(self, input: PredictReviewInput) -> PredictReviewOutput:
        if not self.model:
            raise RuntimeError
        prediction = predict_fn(input_data=input.review, model=self.model)
        return PredictReviewOutput(**prediction)


classification_model = ReviewClassification()


@app.on_event("startup")
def load_artifacts():
    classification_model.load_model()
    print("Ready for inference!")


@app.get("/", tags=["General"], status_code=status.HTTP_200_OK)
def _index(request: Request):
    """Health check."""
    response = {
        "data": "Everything is working as exepected",
    }
    return response


@app.get("/params/{param}", tags=["Parameters"], status_code=status.HTTP_200_OK)
def _param(param: str = Path(...)) -> Dict:
    """Get a specific parameter's value about the model or the training process"""
    if param not in classification_model.properties:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"this parameter {param} is not found ",
        )
    return {param: classification_model.artifacts.get(param)}


@app.post("/predict", tags=["Prediction"], status_code=status.HTTP_200_OK)
async def _predict(
    output: PredictReviewOutput = Depends(classification_model.predict),
) -> PredictReviewOutput:
    """Predict if a review is positive, negative or neutral"""
    # Predict
    return output
