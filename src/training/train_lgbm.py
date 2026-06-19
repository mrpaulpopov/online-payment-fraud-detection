import logging
import time
from src.models.lightgbm_model import lightgbm_model
import lightgbm as lgb
import mlflow
from src.paths import LGBM_MODEL_PATH


def prepare_data_for_lgbm(X_train, X_val, X_test, y_train, y_val, y_test):
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns  # X_train as a gold standard
    X_train[categorical_cols] = X_train[categorical_cols].astype("category")
    X_val[categorical_cols] = X_val[categorical_cols].astype("category")
    X_test[categorical_cols] = X_test[categorical_cols].astype("category")

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    return train_data, valid_data, test_data, X_train, X_val, X_test


def training_lgbm(train_data, valid_data, lgbm_params):
    start = time.time()
    model = lightgbm_model(train_data, valid_data, lgbm_params)

    mlflow.log_params(lgbm_params)  # logging hyperparameters
    mlflow.lightgbm.log_model(model, name="model")
    mlflow.log_metric("best_iteration", model.best_iteration)
    logging.info(f"Training completed in {time.time() - start:.4f}s")

    model.save_model(LGBM_MODEL_PATH, num_iteration=model.best_iteration)
    logging.info(f"Model saved to {LGBM_MODEL_PATH}")
    return


