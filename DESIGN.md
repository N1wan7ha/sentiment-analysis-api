# Sentiment Analysis API – Design Write-Up

## Overview

This project implements a **Sentiment Analysis REST API** using Python, FastAPI, and a trained machine learning model.
The system allows users to send text input and receive a predicted sentiment along with a confidence score.

The architecture follows a **machine learning microservice pattern**, where a trained model is deployed behind a REST API for real-time inference.

---

## Dataset

The model was trained using the **IMDb Movie Reviews Dataset**, which contains **50,000 labeled movie reviews** classified as positive or negative.

This dataset is commonly used for benchmarking sentiment analysis models because it contains balanced and realistic user-generated reviews.

---

## Data Processing

Text data is processed using the following steps:

1. **Tokenization and vectorization** using TF-IDF
2. Removal of common English stop words
3. Limiting features to the **top 10,000 terms** to reduce dimensionality

This converts text into numerical features suitable for machine learning models.

---

## Model Selection

The chosen model is **Logistic Regression**, combined with a **TF-IDF feature extractor**.

Reasons for this choice:

* Efficient for text classification
* Fast training time
* Good baseline performance
* Lightweight and easy to deploy

The model is implemented using **Scikit-learn** with a pipeline combining preprocessing and classification.

---

## Model Evaluation

The model is evaluated using a train/test split (80/20).

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score

The model achieved an approximate accuracy of **~88%** on the test dataset.

---

## API Design

The system exposes three REST endpoints:

### Health Check

```
GET /health
```

Returns:

```
{"status": "ok"}
```

Used to verify that the API service is running.

---

### Single Prediction

```
POST /predict
```

Input:

```
{"text": "This movie was amazing"}
```

Output:

```
{
 "text": "This movie was amazing",
 "sentiment": "positive",
 "confidence": 0.92
}
```

---

### Batch Prediction (Bonus)

```
POST /predict/batch
```

Allows multiple texts to be analyzed in a single request, improving efficiency when processing large volumes of text.

---

## System Architecture

```
Client
   ↓
FastAPI REST API
   ↓
ML Model (TF-IDF + Logistic Regression)
   ↓
Prediction Response (JSON)
```

The trained model is loaded once when the API server starts to minimize inference latency.

---

## Future Improvements

Potential improvements include:

* Using transformer models such as **BERT** for improved accuracy
* Adding **neutral sentiment classification**
* Deploying the API using **Docker**
* Adding **authentication and rate limiting**
* Implementing **model monitoring**

---

## Conclusion

This project demonstrates how a trained machine learning model can be integrated into a production-style API service.
The system provides a scalable approach to performing sentiment analysis through a RESTful interface.
