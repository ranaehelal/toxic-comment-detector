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

from src.model import ToxicCommentClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Toxic Comment Classifier API",
    description="API for classifying toxic comments using LSTM neural network",
    version="1.0.0"
)

# Initialize classifier
classifier = ToxicCommentClassifier()


# Request/Response models
class CommentRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5


class CommentResponse(BaseModel):
    text: str
    is_toxic: bool
    predictions: Dict[str, float]
    positive_labels: List[Dict[str, float]]


class BatchCommentRequest(BaseModel):
    texts: List[str]
    threshold: Optional[float] = 0.5


class BatchCommentResponse(BaseModel):
    results: List[CommentResponse]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models and tokenizer on startup."""
    try:
        classifier.load_model_and_tokenizer()
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
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


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": classifier.model is not None}


# Single prediction endpoint
@app.post("/predict", response_model=CommentResponse)
async def predict_comment(request: CommentRequest):
    """
    Predict toxicity for a single comment.

    Args:
        request: CommentRequest with text and optional threshold

    Returns:
        CommentResponse with prediction results
    """
    try:
        # Get prediction
        result = classifier.predict(request.text, request.threshold)

        # Format positive labels
        positive_labels = [
            {"label": label, "probability": prob}
            for label, prob in result['positive_labels']
        ]

        return CommentResponse(
            text=request.text,
            is_toxic=result['is_toxic'],
            predictions=result,
            positive_labels=positive_labels
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Batch prediction endpoint
@app.post("/predict_batch", response_model=BatchCommentResponse)
async def predict_comments_batch(request: BatchCommentRequest):
    """
    Predict toxicity for multiple comments.

    Args:
        request: BatchCommentRequest with list of texts and optional threshold

    Returns:
        BatchCommentResponse with list of prediction results
    """
    try:
        results = []

        for text in request.texts:
            # Get prediction
            result = classifier.predict(text, request.threshold)

            # Format positive labels
            positive_labels = [
                {"label": label, "probability": prob}
                for label, prob in result['positive_labels']
            ]

            results.append(CommentResponse(
                text=text,
                is_toxic=result['is_toxic'],
                predictions=result,
                positive_labels=positive_labels
            ))

        return BatchCommentResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# Model info endpoint
@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded models."""
    if classifier.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "LSTM",
        "labels": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "vocab_size": 20000,
        "max_sequence_length": 150,
        "model_path": classifier.model_path,
        "tokenizer_path": classifier.tokenizer_path
    }


# Statistics endpoint
@app.get("/stats")
async def get_prediction_stats():
    """Get basic statistics about predictions (placeholder for future implementation)."""
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