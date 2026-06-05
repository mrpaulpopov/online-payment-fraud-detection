import gc
import json
import logging
import os
import sys

import lightgbm as lgb
import mlflow
import pandas as pd
import yaml

from src.data.data_loader import load_data
from src.data.split import train_split
from src.training.evaluation import evaluate_and_log_metrics, cross_validation, plot_shap_values
from src.training.train_lgbm import prepare_data_for_lgbm, find_best_threshold, training_lgbm
from src.training.train_nn import pytorch_preprocessing, training_nn, get_anomaly_scores, assign_anomaly_scores
from src.paths import INFERENCE_PATH, CONFIG_PATH, CACHE_DIR, LGBM_MODEL_PATH

# TODO: temp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)


# ORCHESTRATION
def training_pipeline():
    logging.info('Starting training pipeline')

    # Loading parameters
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    params = config["lgbm_params"]


    # Loading data
    X, y = load_data(table_name='train_final_features')
    X_train, y_train, X_val, y_val, X_test, y_test = train_split(X, y, train_size=0.7, val_size=0.15)

    del X, y
    gc.collect()

    # PyTorch side
    X_train_nn, X_val_nn, X_test_nn = pytorch_preprocessing(X_train, X_val, X_test, y_train, y_val, y_test, high_cardinality_threshold=100)
    logging.info('Starting PyTorch training')

    # For training, we need only isFraud=0 columns
    fraud0_mask = (y_train == 0)
    X_train_nn_short = X_train_nn[fraud0_mask]

    model_autoencoder = training_nn(X_train_nn_short, X_val_nn, X_test_nn, LEARNING_RATE=0.0005, BATCH_SIZE=2048, N_EPOCHS=5)

    # PyTorch Inference
    train_scores = get_anomaly_scores(model_autoencoder, X_train_nn)
    val_scores = get_anomaly_scores(model_autoencoder, X_val_nn)
    test_scores = get_anomaly_scores(model_autoencoder, X_test_nn)
    X_train, X_val, X_test = assign_anomaly_scores(X_train, X_val, X_test, train_scores, val_scores, test_scores)

    del X_train_nn_short, X_train_nn, X_val_nn, X_test_nn, train_scores, val_scores, test_scores
    gc.collect()

    # LightGBM side
    logging.info('Starting LightGBM training')
    train_data, valid_data, test_data, X_train, X_val, X_test = prepare_data_for_lgbm(X_train, X_val, X_test, y_train, y_val, y_test)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run() as run:
        training_lgbm(train_data, valid_data, params)
        cross_validation(X_train, X_val, y_train, y_val, params, n_splits=5)

        # Saving Data
        X_train.assign(isFraud=y_train.values).to_parquet("train_enriched.parquet")
        mlflow.log_artifact("train_enriched.parquet", artifact_path="datasets")
        X_val.assign(isFraud=y_val.values).to_parquet("val_enriched.parquet")
        mlflow.log_artifact("val_enriched.parquet", artifact_path="datasets")
        X_test.assign(isFraud=y_test.values).to_parquet("test_enriched.parquet")
        mlflow.log_artifact("test_enriched.parquet", artifact_path="datasets")

        # Saving Run ID
        run_id = run.info.run_id
        inference_meta = {
            "run_id": run_id,
        }
        INFERENCE_PATH.write_text(json.dumps(inference_meta, indent=4), encoding="utf-8")

    logging.info('Training pipeline finished')



def evaluation_pipeline():
    logging.info('Starting evaluation pipeline')

    # Paths, configs
    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    run_id = inference_meta["run_id"]
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    target_precision = config["business_targets"]["target_precision"]
    target_fpr = config["business_targets"]["target_fpr"]

    # Loading data
    client = mlflow.MlflowClient()
    local_path = client.download_artifacts(run_id, "datasets/train_enriched.parquet", str(CACHE_DIR))
    train_df = pd.read_parquet(local_path)
    local_path = client.download_artifacts(run_id, "datasets/val_enriched.parquet", str(CACHE_DIR))
    val_df = pd.read_parquet(local_path)
    local_path = client.download_artifacts(run_id, "datasets/test_enriched.parquet", str(CACHE_DIR))
    test_df = pd.read_parquet(local_path)

    X_train = train_df.drop(columns=['isFraud'])
    y_train = train_df['isFraud']
    X_val = val_df.drop(columns=['isFraud'])
    y_val = val_df['isFraud']
    X_test = test_df.drop(columns=['isFraud'])
    y_test = test_df['isFraud']

    del train_df, val_df, test_df
    gc.collect()

    # Light data preparation
    train_data, valid_data, test_data, X_train, X_val, X_test = prepare_data_for_lgbm(X_train, X_val, X_test, y_train, y_val, y_test)
    # Loading previously saved model
    model_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_id=run_id):
        logging.info(f'Successfully reconnected to MLflow run: {run_id}')
        logging.info('Starting find_best_threshold')
        best_threshold = find_best_threshold(model_lgbm, X_val, y_val, target_precision)
        logging.info('Starting evaluate train')
        evaluate_and_log_metrics(model_lgbm, X_train, y_train, best_threshold, target_fpr, prefix='train')
        logging.info('Starting evaluate val')
        evaluate_and_log_metrics(model_lgbm, X_val, y_val, best_threshold, target_fpr, prefix='val')
        evaluate_and_log_metrics(model_lgbm, X_test, y_test, best_threshold, target_fpr, prefix='test') # Test

    plot_shap_values(model_lgbm, X_val)


training_pipeline()
# evaluation_pipeline()

# mlflow ui
# http://127.0.0.1:5001