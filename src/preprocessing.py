"""
Text preprocessing functions for the NLP sentiment classifier.
"""
import re
import string
import emoji
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from src.utils import SLANG_DICT, EMOJI_KEEP_DICT, IMPORTANT_STOPWORDS, VOCAB_SIZE, MAX_LEN, save_tokenizer

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English models: python -m spacy download en_core_web_sm")
    nlp = None

def replace_slang(text, slang_map=SLANG_DICT):
    for slang, correct in slang_map.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', correct, text)
    return text


def smart_emoji_handler(text,emoji_keep_dict=EMOJI_KEEP_DICT):
    cleaned = ''
    for char in text:
        if char in emoji_keep_dict:
            cleaned += ' ' + emoji_keep_dict[char] + ' '
        elif char in emoji.EMOJI_DATA:
            continue
        else:
            cleaned += char
    return re.sub(r'\s+', ' ', cleaned).strip()

def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]


def preprocessing_text(text):
     # convert emojis
    text = smart_emoji_handler(text)

    #lowercase
    text=text.lower()

    # use re to remove urls and html and more
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove wiki formatting, special templates
    text = re.sub(r'\{\{.*?}}', '', text)
    text = re.sub(r'\[\[.*?]]', '', text)


    text = replace_slang(text, SLANG_DICT)

     #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'[^a-zA-Z\s!?]', '', text)  # keep only letters + ! and ?


    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # spaCy NLP pipeline: lemmatize, remove stopwords, filter short tokens
    if nlp is not None:
         doc = nlp(text)
         tokens = []

         for token in doc:
             word = token.text.lower()
             # Keep important stopwords or remove regular stopwords
             if word in IMPORTANT_STOPWORDS or (not token.is_stop and len(word) > 2):
                 tokens.append(token.lemma_)

         return " ".join(tokens)

    else:
        # Fallback if spaCy is not available
        words = text.split()
        return " ".join([word for word in words if len(word) > 2])


def create_tokenizer(texts, vocab_size=VOCAB_SIZE):
    """Create and fit a tokenizer on the texts."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

import pandas as pd

def texts_to_sequences(texts, tokenizer, max_len=MAX_LEN):
    """Convert texts to padded sequences."""

    # Ensure input is a pandas Series to support fillna and astype
    if isinstance(texts, list):
        texts = pd.Series(texts)
    texts = texts.fillna("").astype(str)

    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')


def preprocess_dataframe(df, text_column='comment_text', save_path=None):
    tqdm.pandas()

    df_copy = df.copy()

    # Apply preprocessing
    print("Preprocessing text...")
    df_copy['clean_text'] = df_copy[text_column].progress_apply(preprocessing_text)

    # Save if path provided
    if save_path:
        df_copy.to_csv(save_path, index=False)
        print(f"Preprocessed dataframe saved to {save_path}")

    return df_copy



def prepare_training_data(train_df, test_df, label_cols, tokenizer_path="models/tokenizer.pkl"):


    # Preprocess text
    print("Preprocessing training data...")
    train_df = preprocess_dataframe(train_df, save_path="data/train_cleaned.csv")

    print("Preprocessing test data...")
    test_df = preprocess_dataframe(test_df, save_path="data/test_cleaned.csv")

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(train_df['clean_text'])
    save_tokenizer(tokenizer, tokenizer_path)

    # Convert to sequences
    print("Converting to sequences...")
    X_train = texts_to_sequences(train_df['clean_text'], tokenizer)
    X_test = texts_to_sequences(test_df['clean_text'], tokenizer)

    # Prepare labels
    y_train = train_df[label_cols].values

    return X_train, X_test, y_train, tokenizer