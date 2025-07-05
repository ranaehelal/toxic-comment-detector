import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .preprocessing import preprocessing_text
from .utils import VOCAB_SIZE, MAX_LEN, LABEL_COLS, load_tokenizer


def build_lstm_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, embedding_dim=128, lstm_units=64):
    """Improved LSTM model for toxic comment classification."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(LABEL_COLS), activation='sigmoid')  # Multi-label classification
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def train_model(model, X_train, y_train, validation_split=0.1, epochs=4, batch_size=128,
                model_path="models/toxic_lstm_model.h5"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Split training data
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train_part, y_train_part,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Model saved to {model_path}")
    return history


def load_trained_model(model_path="models/toxic_lstm_model.h5"):
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_toxicity(model, tokenizer, text, threshold=0.5):
    cleaned_text = preprocessing_text(text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    predictions = model.predict(padded)[0]
    results = {}
    positive_labels = []

    for i, label in enumerate(LABEL_COLS):
        prob = float(predictions[i])
        results[label] = prob

        if prob >= threshold:
            positive_labels.append((label, prob))

    results['positive_labels'] = positive_labels
    results['is_toxic'] = len(positive_labels) > 0

    return results


class ToxicCommentClassifier:
    def __init__(self, model=None, tokenizer=None):
        self.labels = LABEL_COLS
        if model is None:
            self.model = load_trained_model()
        else:
            self.model = model

        if tokenizer is None:
            self.tokenizer = load_tokenizer()
        else:
            self.tokenizer = tokenizer

    def load_model_and_tokenizer(self):
        self.model = load_trained_model()
        self.tokenizer = load_tokenizer()

    def predict(self, comment, threshold=0.5):
        from .preprocessing import preprocessing_text, texts_to_sequences

        clean = preprocessing_text(comment)
        sequence = texts_to_sequences([clean], self.tokenizer)
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        prediction = self.model.predict(padded)[0]

        is_toxic = np.any(prediction >= threshold)

        positive_labels = [(label, float(prob))
                           for label, prob in zip(LABEL_COLS, prediction) if prob >= threshold]

        return {
            "is_toxic": is_toxic,
            "predictions": {label: float(prob) for label, prob in zip(LABEL_COLS, prediction)},
            "positive_labels": positive_labels
        }

