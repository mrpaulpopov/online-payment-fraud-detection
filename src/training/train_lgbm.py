import json
import logging
import os
import time
from src.models.lightgbm_model import lightgbm_model
import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve
)
from src.paths import INFERENCE_PATH, LGBM_MODEL_PATH
from src.data.split import train_split


def prepare_data_for_lgbm(X_train, X_val, X_test, y_train, y_val, y_test):
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns  # X_train as a gold standard
    X_train[categorical_cols] = X_train[categorical_cols].astype("category")
    X_val[categorical_cols] = X_val[categorical_cols].astype("category")
    X_test[categorical_cols] = X_test[categorical_cols].astype("category")

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    return train_data, valid_data, test_data, X_train, X_val, X_test


def training_lgbm(train_data, valid_data, params):
    start = time.time()
    model = lightgbm_model(train_data, valid_data, params)

    mlflow.log_params(params)  # logging hyperparameters
    mlflow.lightgbm.log_model(model, name="model")
    mlflow.log_metric("best_iteration", model.best_iteration)
    logging.info(f"Training completed in {time.time() - start:.4f}s")

    model.save_model(LGBM_MODEL_PATH, num_iteration=model.best_iteration)
    logging.info(f"Model saved to {LGBM_MODEL_PATH}")
    return


def find_best_threshold(model, X_val, y_val, target_precision, run_id):
    # Threshold управляет переводом из probability 0.0-1.0 в decision 0-1 (not fraud, legit / fraud, to block).
    # С какого момента probability считается fraud?

    # Это автоматический способ нахождения threshold через business target (желаемый TARGET_PRECISION)
    client = mlflow.MlflowClient()
    y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)  # probability from 0.0 to 1.0
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)  # 'меню' всех возможных вариантов

    pr_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precisions[:-1],  # all but scikit last element
        'recall': recalls[:-1]  # all but scikit last element
    })

    # Оставляем только те строки, где Precision >= моего заданного значения
    good_precisions = pr_df[pr_df['precision'] >= target_precision]

    if not good_precisions.empty:
        # Сортируем по Recall по убыванию и берем самую первую строку (где Recall максимальный)
        best_row = good_precisions.sort_values(by='recall', ascending=False).iloc[0]
        best_threshold = best_row['threshold']
        logging.info(f"Target Precision {target_precision} is reachable. Threshold: {best_threshold}")
    else:
        best_threshold = 0.5  # fallback
        logging.warning("Target Precision is unreachable. Using default threshold 0.5.")

    client.log_param(run_id, "best_threshold", best_threshold)
    client.log_param(run_id, "target_precision", target_precision)  # customized parameter

    # "Append" JSON: Read-Append-Overwrite
    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    inference_meta["best_threshold"] = float(best_threshold)
    INFERENCE_PATH.write_text(json.dumps(inference_meta, indent=4), encoding="utf-8")

    return best_threshold
