import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- Download NLTK Resources ----------------
nltk.download('wordnet')
nltk.download('stopwords')

# ---------------- Logger Configuration ----------------
logger = logging.getLogger("text_preprocessing")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# ---------------- Text Preprocessing Functions ----------------
def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([ch for ch in text if not ch.isdigit()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    return re.sub('\s+', ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting text normalization...")
    try:
        df['content'] = df['content'].astype(str)
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logger.info("Text normalization complete.")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def remove_small_sentences(df: pd.DataFrame, col: str = "content") -> pd.DataFrame:
    logger.info("Removing short sentences with fewer than 3 words.")
    df[col] = df[col].apply(lambda x: x if len(x.split()) >= 3 else np.nan)
    return df

def save_processed_data(df_train: pd.DataFrame, df_test: pd.DataFrame, save_dir: str) -> None:
    try:
        logger.info(f"Saving processed data to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        df_train.to_csv(os.path.join(save_dir, "train_processed.csv"), index=False)
        df_test.to_csv(os.path.join(save_dir, "test_processed.csv"), index=False)
        logger.info("Files saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save processed files: {e}")
        raise

# ---------------- Main Pipeline ----------------
def main():
    try:
        logger.info("Reading input CSV files...")
        train_df = pd.read_csv("./data/raw/train.csv")
        test_df = pd.read_csv("./data/raw/test.csv")

        logger.info("Preprocessing train dataset...")
        train_df = normalize_text(train_df)
        train_df = remove_small_sentences(train_df)

        logger.info("Preprocessing test dataset...")
        test_df = normalize_text(test_df)
        test_df = remove_small_sentences(test_df)

        save_path = os.path.join("data", "interim")
        save_processed_data(train_df, test_df, save_path)

        logger.info("Text preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
