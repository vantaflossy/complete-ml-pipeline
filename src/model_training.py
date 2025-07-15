import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        logger.debug("Data loaded successfully")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    """
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatch between number of samples in features and labels.")

        logger.debug("Initializing Random Forest model with parameters: %s", params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug("Model training started with %d samples", x_train.shape[0])
        clf.fit(x_train, y_train)
        logger.debug("Model training completed successfully")

        return clf
    except ValueError as e:
        logger.error(f"Value error during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def save_model(model, model_path: str):
    try:   
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved to {model_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found while saving model: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")
        raise

    
def main():
    """
    Main function to execute the model training pipeline.
    """
    try:
        params = {'n_estimators': 25, 'random_state': 2}
        train_data = load_data('data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(x_train, y_train, params)

        model_path = 'models/model.pkl'
        save_model(clf, model_path)
        logger.info(f"Model saved to {model_path}")    
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()