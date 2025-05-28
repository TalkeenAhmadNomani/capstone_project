import os
import json
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logger

# Load environment variables
load_dotenv()
dagshub_token = os.getenv("CAPSTONE_TEST")

if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set in .env file")

# Set MLflow credentials for DagsHub
# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(repo_owner='TalkeenAhmadNomani', repo_name='capstone_project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/TalkeenAhmadNomani/capstone_project.mlflow')


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('CSV parse error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logger.info('Evaluation metrics computed')
        return metrics
    except Exception as e:
        logger.error('Error during evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump({'run_id': run_id, 'model_path': model_path}, file, indent=4)
        logger.info('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving model info: %s', e)
        raise


def main():
    mlflow.set_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')

            # Log to MLflow
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            if hasattr(clf, 'get_params'):
                for param_name, param_value in clf.get_params().items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(clf, "model")
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logger.error('Model evaluation process failed: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
