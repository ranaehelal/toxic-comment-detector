"""
FastAPI application for toxic comment classification.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from src.model import ToxicCommentClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Toxic Comment Classifier API",
    description="API for classifying toxic comments using LSTM neural network",
    version="1.0.0"
)

# Allow frontend (React) to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # غيريها لاحقاً للدومين الفعلي لو حبيتي
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier
classifier = ToxicCommentClassifier()

# Request/Response models
class CommentRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5

class LabelPrediction(BaseModel):
    label: str
    probability: float

class CommentResponse(BaseModel):
    text: str
    is_toxic: bool
    predictions: Dict[str, float]
    positive_labels: List[LabelPrediction]

class BatchCommentRequest(BaseModel):
    texts: List[str]
    threshold: Optional[float] = 0.5

class BatchCommentResponse(BaseModel):
    results: List[CommentResponse]


@app.get("/")
async def root():
    return {
        "message": "Toxic Comment Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": classifier.model is not None}


@app.post("/predict", response_model=CommentResponse)
async def predict_comment(request: CommentRequest):
    try:
        result = classifier.predict(request.text, request.threshold)

        predictions_dict = result["predictions"]

        positive_labels = [
            {"label": label, "probability": prob}
            for label, prob in result['positive_labels']
        ]

        return CommentResponse(
            text=request.text,
            is_toxic=result['is_toxic'],
            predictions=predictions_dict,
            positive_labels=positive_labels
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=BatchCommentResponse)
async def predict_comments_batch(request: BatchCommentRequest):
    try:
        results = []

        for text in request.texts:
            result = classifier.predict(text, request.threshold)

            predictions_dict = result["predictions"]

            positive_labels = [
                {"label": label, "probability": prob}
                for label, prob in result['positive_labels']
            ]

            results.append(CommentResponse(
                text=text,
                is_toxic=result['is_toxic'],
                predictions=predictions_dict,
                positive_labels=positive_labels
            ))

        return BatchCommentResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model_info")
async def get_model_info():
    if classifier.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "LSTM",
        "labels": classifier.labels,
        "vocab_size": 20000,
        "max_sequence_length": 150,
        "model_path": getattr(classifier, "model_path", "N/A"),
        "tokenizer_path": getattr(classifier, "tokenizer_path", "N/A")
    }


@app.get("/stats")
async def get_prediction_stats():
    return {
        "message": "Statistics endpoint - implement based on your needs",
        "suggestions": [
            "Add request counter",
            "Track prediction distributions",
            "Monitor response times",
            "Log toxic vs clean ratios"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
