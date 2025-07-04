import sys
import os
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data, LABEL_COLS, load_tokenizer
from src.model import load_trained_model, ToxicCommentClassifier
from src.preprocessing import texts_to_sequences, preprocess_dataframe


def evaluate_model(model, tokenizer, X_test, y_test, threshold=0.5):
    """Evaluate the model on test data."""
    print("Making predictions...")
    predictions = model.predict(X_test)

    # Convert probabilities to binary predictions
    binary_predictions = (predictions >= threshold).astype(int)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"Threshold: {threshold}")
    print(f"Test samples: {len(y_test)}")

    for i, label in enumerate(LABEL_COLS):
        if np.sum(y_test[:, i]) > 0:
            auc = roc_auc_score(y_test[:, i], predictions[:, i])
            accuracy = accuracy_score(y_test[:, i], binary_predictions[:, i])

            print(f"\n{label.upper()}:")
            print(f"  AUC: {auc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Positive examples: {np.sum(y_test[:, i])}")

    print("\n" + "=" * 50)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 50)

    report = classification_report(
        y_test, binary_predictions,
        target_names=LABEL_COLS,
        zero_division=0
    )
    print(report)

    return predictions, binary_predictions


def test_sample_comments(tokenizer, model):
    """Test the model on sample comments."""
    print("\n" + "=" * 50)
    print("TESTING SAMPLE COMMENTS")
    print("=" * 50)

    classifier = ToxicCommentClassifier(model=model, tokenizer=tokenizer)

    test_comments = [
        "I love this product! Great quality and service.",
        "This is absolutely terrible, worst experience ever.",
        "You're an idiot and should go to hell!",
        "Thank you for your help, much appreciated.",
        "I hate you so much, you stupid moron!",
        "This looks amazing, can't wait to try it!",
        "Kill yourself, nobody wants you here.",
        "Great job on this project, well done!",
    ]

    for comment in test_comments:
        result = classifier.predict(comment)
        print(f"\nComment: '{comment}'")
        print(f"Toxic: {'YES' if result['is_toxic'] else 'NO'}")

        if result['is_toxic']:
            print("Labels detected:")
            for label, prob in result['positive_labels']:
                print(f"  - {label}: {prob:.3f}")


def main():
    print("Starting evaluation...")

    # Step 1: Load test data - Fixed path consistency
    print("Loading test data...")
    try:
        test_df = pd.read_csv("data/test_cleaned.csv")
        print("Loaded preprocessed test data.")
    except:
        print("Preprocessed file not found. Loading and preprocessing raw data...")
        _, test_df, _ = load_data("data/")  # Fixed: consistent path format
        if test_df is None:
            print("Failed to load raw data. Please check data directory.")
            return
        test_df = preprocess_dataframe(test_df, save_path="data/test_cleaned.csv")

    # Step 2: Load test labels
    print("Loading real test labels...")
    try:
        test_labels_df = pd.read_csv("data/test_labels.csv")
    except FileNotFoundError:
        print("test_labels.csv not found in data directory.")
        return

    # Step 3: Merge and filter
    merged_df = test_df.merge(test_labels_df, on="id")
    filtered_df = merged_df[(merged_df[LABEL_COLS] != -1).all(axis=1)].reset_index(drop=True)

    print(f"Filtered test set size: {filtered_df.shape[0]}")
    print("Positive labels count:")
    print(filtered_df[LABEL_COLS].sum())

    # Step 4: Load model and tokenizer
    print("Loading model and tokenizer...")
    model = load_trained_model()
    tokenizer = load_tokenizer()

    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer.")
        print("Make sure you have:")
        print("- models/toxic_lstm_model.h5")
        print("- models/tokenizer.pkl")
        return

    # Step 5: Prepare data
    print("Preparing test sequences...")
    X_test = texts_to_sequences(filtered_df['clean_text'], tokenizer)
    y_test = filtered_df[LABEL_COLS].values

    # Step 6: Evaluation
    evaluate_model(model, tokenizer, X_test, y_test)

    # Step 7: Predict sample comments
    test_sample_comments(tokenizer, model)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()