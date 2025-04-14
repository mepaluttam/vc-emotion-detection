import pandas as pd
import numpy as np
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ---------------- Logger Setup ----------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# ---------------- Utility Functions ----------------

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_path: str):
    try:
        test_df = pd.read_csv(test_path)
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        logger.info(f"Test data loaded from {test_path}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }

        logger.info("Model evaluation completed.")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, output_path: str):
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise

# ---------------- Main Pipeline ----------------

def main():
    try:
        model_path = "./models/model.pkl"
        test_data_path = "./data/processed/test_tfidf.csv"
        metrics_output_path = "./reports/metric.json"

        model = load_model(model_path)
        X_test, y_test = load_test_data(test_data_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_output_path)

        logger.info("Model evaluation pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()
