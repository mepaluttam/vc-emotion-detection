import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- Logger Configuration ----------------
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# ---------------- Utility Functions ----------------
def load_params(path: str) -> int:
    try:
        params = yaml.safe_load(open(path, 'r'))
        max_features = params['feature_engineering']['max_features']
        logger.info(f"Loaded max_features = {max_features} from {path}")
        return max_features
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise

def load_processed_data(train_path: str, test_path: str):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info("Processed training and test data loaded successfully.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def apply_bow(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int):
    try:
        logger.info("Applying Bag-of-Words (CountVectorizer)...")
        train_df['content'] = train_df['content'].fillna("")
        test_df['content'] = test_df['content'].fillna("")

        X_train = train_df['content'].values
        y_train = train_df['sentiment'].values

        X_test = test_df['content'].values
        y_test = test_df['sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_features = pd.DataFrame(X_train_bow.toarray())
        test_features = pd.DataFrame(X_test_bow.toarray())

        train_features['label'] = y_train
        test_features['label'] = y_test

        logger.info("Bag-of-Words transformation completed.")
        return train_features, test_features
    except Exception as e:
        logger.error(f"Error during BoW transformation: {e}")
        raise

def save_feature_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: str):
    try:
        os.makedirs(save_path, exist_ok=True)
        train_df.to_csv(os.path.join(save_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(save_path, "test_bow.csv"), index=False)
        logger.info(f"Feature-engineered data saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save feature files: {e}")
        raise

# ---------------- Main Pipeline ----------------
def main():
    try:
        # Paths
        train_path = "./data/interim/train_processed.csv"
        test_path = "./data/interim/test_processed.csv"
        save_path = os.path.join("data", "processed")
        params_path = "params.yaml"

        # Load config and data
        max_features = load_params(params_path)
        train_df, test_df = load_processed_data(train_path, test_path)

        # Feature engineering
        train_bow, test_bow = apply_bow(train_df, test_df, max_features)

        # Save feature-engineered data
        save_feature_data(train_bow, test_bow, save_path)

        logger.info("Feature engineering pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Feature engineering pipeline failed: {e}")

if __name__ == "__main__":
    main()
