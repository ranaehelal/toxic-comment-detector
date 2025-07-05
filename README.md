# Toxic Comment Classifier

A machine learning system for detecting toxic comments using LSTM neural networks with FastAPI backend and React frontend.

## 🚀 Features

- **Multi-label Classification**: Detects 6 types of toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **REST API**: FastAPI backend for real-time predictions
- **Interactive Frontend**: React web interface with visualization
- **Batch Processing**: Analyze multiple comments at once

## 📁 Project Structure

```
toxic-comment-classifier/
├── app/main.py                 # FastAPI application
├── frontend/src/               # React frontend
├── src/                        # ML pipeline (model, preprocessing, utils)
├── scripts/                    # Training and evaluation scripts
├── data/                       # Dataset directory
├── models/                     # Saved models
└── requirements.txt
```

## 🛠️ Installation

### Backend Setup

```bash
git clone https://github.com/ranaehelal/toxic-comment-detector.git
cd toxic-comment-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Frontend Setup

```bash
cd frontend
npm install
```

## 📊 Dataset

Download the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) dataset and extract to `data/` directory.

## 🏋️ Training

```bash
python scripts/train.py
```

The model uses a bidirectional LSTM with:
- Vocabulary size: 20,000
- Max sequence length: 150
- 6 sigmoid outputs for multi-label classification

## 🚀 Running the Application

### Start Backend
```bash
python -m uvicorn app.main:app --reload --port 8000
```

### Start Frontend
```bash
cd frontend
npm start
```

Access at:
- **Web Interface**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 🔧 API Usage

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "You are amazing!", "threshold": 0.5}'
```

### Response
```json
{
  "text": "You are amazing!",
  "is_toxic": false,
  "predictions": {
    "toxic": 0.1,
    "severe_toxic": 0.05,
    "obscene": 0.02,
    "threat": 0.01,
    "insult": 0.03,
    "identity_hate": 0.01
  },
  "positive_labels": []
}
```

## 📈 Evaluation

```bash
python scripts/evaluate.py
```

Expected performance: ~0.85-0.95 AUC-ROC across different toxicity labels.

##  Troubleshooting

- **Model not found**: Run `python scripts/train.py` first
- **spaCy error**: Install with `python -m spacy download en_core_web_sm`
- **CORS issues**: Ensure backend runs on port 8000

##  Tech Stack

- **Backend**: FastAPI, TensorFlow, scikit-learn, spaCy
- **Frontend**: React, Chart.js, Bootstrap
- **ML**: Bidirectional LSTM, text preprocessing pipeline

---

