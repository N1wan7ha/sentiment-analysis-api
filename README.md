# Sentiment Analysis API

## Overview

This project implements a **Sentiment Analysis REST API** using **Python, FastAPI, and a Machine Learning model**.
The API predicts whether a given piece of text expresses **positive or negative sentiment** and returns a confidence score.

The system is designed as a simple **ML microservice**, where a trained model is exposed through a REST API for inference.

---

## Features

* Train a sentiment analysis model using the **IMDb dataset**
* Predict sentiment for a **single text input**
* Predict sentiment for **multiple texts (batch endpoint)**
* Health check endpoint for service monitoring
* Interactive API documentation using **Swagger UI**

---

## Tech Stack

* **Python 3**
* **FastAPI** – API framework
* **Scikit-learn** – Machine learning model
* **TF-IDF Vectorizer** – Text feature extraction
* **Logistic Regression** – Sentiment classifier
* **Uvicorn** – ASGI server

---

## Project Structure

```
sentiment-analysis-api
│
├── app
│   ├── main.py        # FastAPI application
│   ├── model.py       # Model loading and prediction
│   └── schemas.py     # Request/response schemas
│
├── model
│   └── sentiment_model.pkl
│
├── train.py           # Model training script
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Model Training

The model is trained using the **IMDb movie reviews dataset**, which contains **50,000 labeled reviews**.

### Pipeline

1. Text preprocessing using **TF-IDF vectorization**
2. Feature extraction with **10,000 max features**
3. Classification using **Logistic Regression**

### Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Typical accuracy achieved: **~0.88**

---

## Installation

Python version: **3.10+**

Clone the repository or unzip the project.

### Create Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

---

## Train the Model

```
python train.py
```

This will:

* Download the IMDb dataset
* Train the model
* Save it to:

```
model/sentiment_model.pkl
```

---

## Run the API

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

The API will run at:

```
http://127.0.0.1:8000
```

---

## API Endpoints

### Health Check

```
GET /health
```

Response:

```
{
  "status": "ok"
}
```

---

### Predict Sentiment

```
POST /predict
```

Request:

```
{
  "text": "This movie was amazing!"
}
```

Response:

```
{
  "text": "This movie was amazing!",
  "sentiment": "positive",
  "confidence": 0.92
}
```

Exact curl example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I absolutely love how fast the delivery was!\"}"
```

---

### Batch Prediction (Bonus)

```
POST /predict/batch
```

Request:

```
{
  "texts": [
    "I love this movie",
    "This film was terrible"
  ]
}
```

Response:

```
[
  {
    "text": "I love this movie",
    "sentiment": "positive",
    "confidence": 0.93
  },
  {
    "text": "This film was terrible",
    "sentiment": "negative",
    "confidence": 0.91
  }
]
```

---

## API Documentation

FastAPI automatically generates documentation at:

```
http://127.0.0.1:8000/docs
```

This allows interactive testing of all endpoints.

---

## Design Decisions

## Approach (Required Write-Up)

I used a TF-IDF + Logistic Regression pipeline because it is a strong and efficient baseline for binary sentiment classification. The IMDb dataset was selected because it is balanced, widely used, and suitable for reproducible benchmarking. The API loads the trained model once on startup to keep inference fast and avoid repeated disk I/O on each request. With more time, I would add a neutral class and evaluate a transformer model (such as DistilBERT) to compare performance against this baseline.

### Why TF-IDF + Logistic Regression?

This combination provides:

* Fast training
* Strong baseline performance for text classification
* Lightweight deployment for APIs

It is commonly used as a **baseline NLP model** before applying more complex deep learning approaches.

---

## Possible Improvements

Future enhancements could include:

* Using **transformer models (BERT / RoBERTa)**
* Adding **neutral sentiment classification**
* Deploying the API using **Docker**
* Adding **rate limiting and authentication**


## Running with Docker

You can run the API using Docker.

### Build the Image

```
docker build -t sentiment-api .
```

### Run the Container

```
docker run -p 8000:8000 sentiment-api
```

The API will be available at:

```
http://localhost:8000
```

Swagger documentation:

```
http://localhost:8000/docs
```
