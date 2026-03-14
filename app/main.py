from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from app.model import load_model, predict_sentiment
from app.schemas import TextRequest, PredictionResponse
from app.schemas import BatchRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts up
    load_model()
    yield
    # Any teardown logic would go here


app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text using ML model",
    version="1.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):

    sentiment, confidence = predict_sentiment(request.text)

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": confidence
    }

@app.post("/predict/batch", response_model=List[PredictionResponse])
def batch_predict(request: BatchRequest):

    if not request.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")

    results = []

    for text in request.texts:
        sentiment, confidence = predict_sentiment(text)

        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        })

    return results