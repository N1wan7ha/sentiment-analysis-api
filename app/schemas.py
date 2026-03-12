from pydantic import BaseModel
from typing import List


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


class BatchRequest(BaseModel):
    texts: List[str]