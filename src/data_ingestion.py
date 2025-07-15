import yaml
import pandas as pd
import os
from sklearn.model_selection import train_test_split
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

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(data_url)
        logger.debug("Data loaded successfully")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.
    """
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug("Data Preprocessing completed")
        return df
    except KeyError as e:
        logger.error(f"Missing Column in Dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        raise

def save_data(train_data: pd.DataFrame, output_path: str, test_data: pd.DataFrame) -> None:
    """
    Save the DataFrame to a CSV file.
    """
    try:
        raw_data_path = os.path.join(output_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'data.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'), index=False)
        logger.debug("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    params = load_params('params.yaml')
    test_size = params['data_ingestion']['test_size']
    try:
        #test_size = 0.2
        data_path = "https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv"
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, output_path='data', test_data=test_data)

    except Exception as e:
        logger.error(f"Failed to complete data ingestion process: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()  
