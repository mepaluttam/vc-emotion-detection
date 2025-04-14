import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

# ---------------- Logger Configuration ----------------
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)

# ---------------- Function Definitions ----------------

def load_params(params_path: str) -> float:
    try:
        logger.info(f"Loading parameters from: {params_path}")
        with open(params_path, 'r') as file:
            config = yaml.safe_load(file)
        test_size = config['data_ingestion']['test_size']
        logger.debug(f"Loaded test_size: {test_size}")
        return test_size
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Reading data from URL: {url}")
        df = pd.read_csv(url)
        logger.debug(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to read data: {e}")
        raise

def processed_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Processing data...")
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.debug(f"Processed data shape: {final_df.shape}")
        return final_df
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        logger.info(f"Saving data to {data_path}...")
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug("Data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        logger.info("Starting data ingestion pipeline...")
        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = processed_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
        logger.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Pipeline failed due to error: {e}")

if __name__ == "__main__":
    main()