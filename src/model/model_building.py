import pandas as pd
import numpy as np
import os
import yaml
import pickle
import logging
from sklearn.ensemble import GradientBoostingClassifier

# ---------------- Logger Setup ----------------
logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# ---------------- Utility Functions ----------------
def load_params(params_path: str):
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)['model_building']
        logger.info(f"Model parameters loaded from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Failed to load params: {e}")
        raise

def load_training_data(train_path: str):
    try:
        train_df = pd.read_csv(train_path)
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        logger.info("Training data loaded successfully.")
        return X_train, y_train
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    try:
        clf = GradientBoostingClassifier(
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators']
        )
        clf.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return clf
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

def save_model(model, model_path: str):
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

# ---------------- Main Pipeline ----------------
def main():
    try:
        train_path = "./data/processed/train_tfidf.csv"
        model_path = "./models/model.pkl"
        params_path = "params.yaml"

        params = load_params(params_path)
        X_train, y_train = load_training_data(train_path)
        model = train_model(X_train, y_train, params)
        save_model(model, model_path)

        logger.info("Model training pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
