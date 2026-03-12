from fastapi import FastAPI
from app.model import load_model, predict_sentiment
from app.schemas import TextRequest, PredictionResponse
from app.schemas import BatchRequest

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text using ML model",
    version="1.0"
)


@app.on_event("startup")
def startup_event():
    load_model()


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

@app.post("/predict/batch")
def batch_predict(request: BatchRequest):

    if not request.texts:
        return {"error": "Text list cannot be empty"}

    results = []

    for text in request.texts:
        sentiment, confidence = predict_sentiment(text)

        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        })

    return results