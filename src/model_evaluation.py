import os
from dvclive import Live # type: ignore
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml

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

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_model(file_path: str):
    """
    Load a trained model from a file.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug("Model loaded successfully")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_data(file_path: str):
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

def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model using test data and return metrics.
    """
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug("Model evaluation completed successfully")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, file_path: str):
    """
    Save evaluation metrics to a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved successfully to %s", file_path)
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

def main():
    """
    Main function to load parameters, model, and test data, evaluate the model, and save metrics.
    """
    try:
        params = load_params('params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, x_test, y_test) # type: ignore

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test)) #type: ignore
            live.log_metric('precision', precision_score(y_test, y_test)) # type: ignore
            live.log_metric('recall', recall_score(y_test, y_test)) # type: ignore

            live.log_params(params)

        save_metrics(metrics, 'metrics/evaluation_metrics.json')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
        

if __name__ == "__main__":
    main()