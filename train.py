"""
train.py

Training script for Sentiment Analysis model.

This script:
1. Downloads the IMDb dataset
2. Preprocesses text using TF-IDF
3. Trains a Logistic Regression classifier
4. Evaluates the model
5. Saves the trained model to disk

Author: Niwantha Sithumal
"""

import pandas as pd
import joblib

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    """Load IMDb dataset"""
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    df = pd.DataFrame(dataset["train"])
    return df


def train_model(X_train, y_train):
    """Create and train the ML pipeline"""

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""

    print("\nEvaluating model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def save_model(model):
    """Save trained model"""

    model_path = "model/sentiment_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel saved at: {model_path}")


def main():

    df = load_data()

    X = df["text"]
    y = df["label"]

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()