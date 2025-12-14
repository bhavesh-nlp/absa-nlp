from typing import List
from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str


class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str


class PredictResponse(BaseModel):
    aspects: List[AspectSentiment]
