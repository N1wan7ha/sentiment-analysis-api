import joblib

from fastapi import HTTPException


MODEL_PATH = "model/sentiment_model.pkl"

model = None


def load_model():
    global model
    model = joblib.load(MODEL_PATH)
    return model


def predict_sentiment(text: str):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure train.py has been run and the server has started correctly."
        )

    prediction = model.predict([text])[0]

    probability = model.predict_proba([text])[0].max()

    sentiment_map = {
        0: "negative",
        1: "positive"
    }

    sentiment = sentiment_map[prediction]

    return sentiment, float(probability)