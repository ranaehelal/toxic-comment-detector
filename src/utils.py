"""
Utility functions and constants for the NLP sentiment classifier project.
"""
import kagglehub
import pandas as pd
import pickle
import os
SLANG_DICT = {
    # Common informal
    "u": "you",
    "ur": "your",
    "r": "are",
    "ya": "you",
    "gonna": "going_to",
    "wanna": "want_to",
    "gotta": "got_to",
    "lemme": "let_me",
    "kinda": "kind_of",
    "ain’t": "is_not",
    "y'all": "you_all",
    "nah": "no",
    "thx": "thanks",
    "pls": "please",
    "plz": "please",

    # Emotions/reactions
    "lol": "laugh",
    "lmao": "laugh",
    "rofl": "laugh",
    "omg": "oh_my_god",
    "wtf": "what_the_fuck",
    "smh": "shaking_my_head",
    "tbh": "to_be_honest",
    "idk": "i_do_not_know",
    "ikr": "i_know_right",
    "bruh": "bro",
    "af": "very",
    "fml": "fuck_my_life",
    "ffs": "for_fucks_sake",

    # Toxic/hateful
    "fu": "fuck_you",
    "fck": "fuck",
    "f*ck": "fuck",
    "f---": "fuck",
    "stfu": "shut_the_fuck_up",
    "kys": "kill_yourself",
    "bitch": "bitch",
    "biatch": "bitch",
    "a**hole": "asshole",
    "d*ck": "dick",
    "nigga": "nigger",
    "n1gga": "nigger",
    "ni99a": "nigger",
    "retard": "retard",
    "r3tard": "retard",
    "re**rd": "retard",
    "sucka": "sucker",

    # Abbreviated hate
    "h8": "hate",
    "h8r": "hater",
    "killurself": "kill_yourself",
    "go2hell": "go_to_hell",
    "gtfo": "get_the_fuck_out",
    "gtf": "get_the_fuck_out",
    "die": "die",
    "idiot": "idiot",
    "moron": "moron",
    "loser": "loser",

    # Mocking
    "lulz": "laugh",
    "noob": "novice",
    "n00b": "novice",
    "scrub": "bad_player",
    "simp": "obsessed_fan",
    "trash": "trash",
    "cringe": "cringe",

    # Threats
    "bomb": "bomb",
    "shoot": "shoot",
    "stab": "stab",
    "burn": "burn",
    "rape": "rape"
}
EMOJI_KEEP_DICT = {
    "☠": "skull",
    "😂": "laugh",
    "⚔": "violence",
    "😜": "sarcasm_face",
    "😉": "wink",
}
IMPORTANT_STOPWORDS = {"not", "no", "never", "n't"}
MAX_LEN = 150
VOCAB_SIZE = 20000
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def download_kaggle_data():
    """Download the Jigsaw Toxic Comment Classification dataset from Kaggle."""
    try:
        path = kagglehub.dataset_download("julian3833/jigsaw-toxic-comment-classification-challenge")
        print(f"Path to dataset files: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def load_data(data_path):
    """Load the training and test datasets."""
    try:
        train_df = pd.read_csv(f"{data_path}/train.csv")
        test_df = pd.read_csv(f"{data_path}/test.csv")
        test_labels_df = pd.read_csv(f"{data_path}/test_labels.csv")

        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Test labels shape: {test_labels_df.shape}")

        return train_df, test_df, test_labels_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def save_tokenizer(tokenizer, filepath="models/tokenizer.pkl"):
    """Save the tokenizer to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {filepath}")


def load_tokenizer(filepath="models/tokenizer.pkl"):
    """Load the tokenizer from a pickle file."""
    try:
        with open(filepath, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {filepath}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


def contains_slang(text, slang_words):
    """Check if text contains any slang words."""
    words = text.lower().split()
    return any(word in slang_words for word in words)