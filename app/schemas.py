from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    review: str = Query(None, min_length=1)


class PredictReview(BaseModel):
    reviews: List[Text]

    @validator("reviews")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "reviews": [{"review": "I am happy."}, {"review": "I am not happy."}]
            }
        }
