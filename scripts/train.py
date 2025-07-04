"""
Training script for the toxic comment classifier.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import download_kaggle_data, load_data, LABEL_COLS
from src.preprocessing import prepare_training_data
from src.model import build_lstm_model, train_model


def main():
    print("Starting training pipeline...")

    # Download and load data
    print("Loading data...")
    #data_path = download_kaggle_data()
    #if data_path is None:
        #print("Failed to download data. Exiting.")
        #return

    train_df, test_df, test_labels_df = load_data('data/')
    if train_df is None:
        print("Failed to load data. Exiting.")
        return

    # Basic data info
    print(f"Training data shape: {train_df.shape}")
    print(f"Label distribution:")
    print(train_df[LABEL_COLS].sum().sort_values(ascending=False))

    # Prepare training data
    print("Preparing training data...")
    X_train, X_test, y_train, tokenizer = prepare_training_data(
        train_df, test_df, LABEL_COLS
    )

    print(f"Training sequences shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Build models
    print("Building models...")
    model = build_lstm_model()
    print(model.summary())

    # Train models
    print("Training models...")
    history = train_model(
        model, X_train, y_train,
        validation_split=0.1,
        epochs=4,
        batch_size=128
    )

    # Print training results
    print("Training completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Test predictions on sample comments
    print("\nTesting predictions on sample comments...")
    test_comments = [
        "I love this product! Absolutely fantastic experience.",
        "You're a genius, keep up the great work!",
        "This is the worst service ever, totally disappointed.",
        "You suck and nobody likes you.",
        "Go to hell, you idiot!",
        "Thank you so much for your support!",
    ]

    from src.model import predict_toxicity

    for comment in test_comments:
        result = predict_toxicity(model, tokenizer, comment)
        print(f"\nComment: {comment}")
        if result['is_toxic']:
            print("Predicted as TOXIC:")
            for label, prob in result['positive_labels']:
                print(f"  - {label}: {prob:.3f}")
        else:
            print("Predicted as CLEAN")


if __name__ == "__main__":
    main()