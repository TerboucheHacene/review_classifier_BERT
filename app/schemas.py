from pathlib import Path
from fastapi import Query
from pydantic import BaseModel, BaseSettings


class PredictReviewInput(BaseModel):
    review: str = Query(None, min_length=1)

    class Config:
        schema_extra = {"example": {"review": "I am happy."}}


class PredictReviewOutput(BaseModel):
    probability: float
    predicted_label: str
    predicted_index: int


class Settings(BaseSettings):
    model_path: Path = "logs/"
    config_file: Path = "logs/config.json"
    MODEL_NAME: str = "model.pth"
