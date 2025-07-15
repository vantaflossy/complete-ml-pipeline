import yaml
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

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

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True)
        logger.debug("Data loaded successfully")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF vectorization to the text column of the DataFrame.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train = train_data['text'].values
        x_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(x_train)
        X_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(X_train_bow.toarray()) # type: ignore
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray()) # type: ignore
        test_df['label'] = y_test

        logger.debug("TF-IDF applied and data transformed")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error in TF-IDF vectorization: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    """
    Main function to execute the feature engineering pipeline.
    """
    params = load_params('params.yaml')
    max_features = params['feature_engineering']['max_features']

    try:
        
        #max_features = 50

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join('data', 'processed', 'train_tfidf.csv'))
        save_data(test_df, os.path.join('data', 'processed', 'test_tfidf.csv'))

        logger.info("Feature engineering completed successfully")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    main()
