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
from src.training.evaluation import evaluate_and_log_metrics, cross_validation, plot_shap_values, find_best_threshold
from src.training.train_lgbm import prepare_data_for_lgbm, training_lgbm
from src.training.train_nn import training_nn, pytorch_anomaly_scores
from src.training.nn_data_preparation import pytorch_preprocessing, assign_anomaly_scores, pytorch_filtering_rows
from src.paths import INFERENCE_PATH, CONFIG_PATH, LGBM_MODEL_PATH, CACHE_DIR

# TODO: temp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)


# ORCHESTRATION
def training_pipeline():
    logging.info('Starting training pipeline')

    # Loading config
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    use_autoencoder = config["pipeline"]["use_autoencoder"]

    # Initializing .json with meta-information
    if not INFERENCE_PATH.exists() or INFERENCE_PATH.stat().st_size == 0:
        INFERENCE_PATH.write_text("{}", encoding="utf-8")

    # Loading data
    X, y, _ = load_data(table_name='train_final_features')
    X_train, y_train, X_val, y_val, X_test, y_test = train_split(X, y, config["train_split"])
    del X, y
    gc.collect()

    if use_autoencoder:
        # ------------ PyTorch side --------------
        X_train_nn, X_val_nn, X_test_nn = pytorch_preprocessing(X_train, X_val, X_test, y_train, config["preprocessing"])
        X_train_nn_short, X_val_nn_short = pytorch_filtering_rows(X_train_nn, X_val_nn, y_train, y_val)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run() as run:
        if use_autoencoder:
            # ------------ PyTorch side --------------
            logging.info('Autoencoder+LGBM pipeline: starting PyTorch training')
            model_autoencoder, pt_val_loss = training_nn(X_train_nn_short, X_val_nn, X_test_nn,
                                                         config["pytorch_params"])
            pt_params = {f"pt_{k}": v for k, v in config["pytorch_params"].items()}
            mlflow.log_params(pt_params)
            mlflow.log_metric("pt_val_loss", pt_val_loss)

            # PyTorch Inference
            train_scores = pytorch_anomaly_scores(model_autoencoder, X_train_nn)
            val_scores = pytorch_anomaly_scores(model_autoencoder, X_val_nn)
            test_scores = pytorch_anomaly_scores(model_autoencoder, X_test_nn)
            X_train, X_val, X_test = assign_anomaly_scores(X_train, X_val, X_test, train_scores, val_scores, test_scores)

            del X_train_nn_short, X_train_nn, X_val_nn, X_test_nn, train_scores, val_scores, test_scores
            gc.collect()
        else:
            logging.info('Baseline LGBM pipeline: skipping PyTorch training')

        # ------------ LightGBM side --------------
        logging.info('Starting LightGBM training')
        train_data, valid_data, test_data, X_train, X_val, X_test = prepare_data_for_lgbm(X_train, X_val, X_test, y_train,
                                                                                      y_val, y_test)

        training_lgbm(train_data, valid_data, config["lgbm_params"])
        lgbm_params = {f"lgbm_{k}": v for k, v in config["lgbm_params"].items()}
        mlflow.log_params(lgbm_params)

        cross_validation(X_train, X_val, y_train, y_val, config["lgbm_params"], n_splits=5)

        # Saving Data
        local_path = str(CACHE_DIR / "train_enriched.parquet")
        X_train.assign(isFraud=y_train.values).to_parquet(local_path)
        mlflow.log_artifact(local_path, artifact_path="datasets")
        local_path = str(CACHE_DIR / "val_enriched.parquet")
        X_val.assign(isFraud=y_val.values).to_parquet(local_path)
        mlflow.log_artifact(local_path, artifact_path="datasets")
        local_path = str(CACHE_DIR / "test_enriched.parquet")
        X_test.assign(isFraud=y_test.values).to_parquet(local_path)
        mlflow.log_artifact(local_path, artifact_path="datasets")

        # Saving Run ID
        # "Append" JSON: Read-Append-Write
        inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
        inference_meta["run_id"] = run.info.run_id
        INFERENCE_PATH.write_text(json.dumps(inference_meta, indent=4), encoding="utf-8")

    logging.info('Training pipeline finished')


def evaluation_pipeline():
    logging.info('Starting evaluation pipeline')

    # Paths, configs
    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    run_id = inference_meta["run_id"]
    logging.info(f'Found run_id: {run_id}')
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    business_target_precision = config["business_targets"]["business_target_precision"]
    threshold_strategy = config["business_targets"]["threshold_strategy"]
    target_fpr = config["business_targets"]["target_fpr"]

    # MLflow settings
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    # Loading data
    local_path = client.download_artifacts(run_id, "datasets/train_enriched.parquet")
    train_df = pd.read_parquet(local_path)
    local_path = client.download_artifacts(run_id, "datasets/val_enriched.parquet")
    val_df = pd.read_parquet(local_path)
    local_path = client.download_artifacts(run_id, "datasets/test_enriched.parquet")
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
    train_data, valid_data, test_data, X_train, X_val, X_test = prepare_data_for_lgbm(X_train, X_val, X_test, y_train,
                                                                                      y_val, y_test)
    # Loading previously saved model
    model_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)

    best_threshold = find_best_threshold(model_lgbm, X_val, y_val, business_target_precision, threshold_strategy,
                                         run_id)
    logging.info('Starting evaluate train, val, test')
    evaluate_and_log_metrics(model_lgbm, X_train, y_train, best_threshold, target_fpr, run_id, prefix='train')
    evaluate_and_log_metrics(model_lgbm, X_val, y_val, best_threshold, target_fpr, run_id, prefix='val')
    evaluate_and_log_metrics(model_lgbm, X_test, y_test, best_threshold, target_fpr, run_id, prefix='test')  # Test

    plot_shap_values(model_lgbm, X_val, run_id)

# mlflow ui
# http://127.0.0.1:5001
