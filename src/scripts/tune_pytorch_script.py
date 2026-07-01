import logging
import sys

import optuna
import lightgbm as lgb
import mlflow
import json
import yaml
from sklearn.metrics import average_precision_score

from src.data.data_loader import load_data
from src.data.split import train_split
from src.paths import CONFIG_PATH
from src.training.train_lgbm import prepare_data_for_lgbm
from src.training.train_nn import training_nn
from src.training.nn_data_preparation import pytorch_preprocessing, pytorch_filtering_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)

# Loading data
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
X, y, _ = load_data(table_name='train_final_features')
X_train, y_train, X_val, y_val, X_test, y_test = train_split(X, y, config["train_split"])
X_train_nn, X_val_nn, X_test_nn = pytorch_preprocessing(X_train, X_val, X_test, y_train, config["preprocessing"])
X_train_nn_short, X_val_nn_short = pytorch_filtering_rows(X_train_nn, X_val_nn, y_train, y_val)

def objective(trial):
    with mlflow.start_run(nested=True):
        print("Trial started.")
        latent_dim = trial.suggest_int("latent_dim", 8, 64)
        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

        trial_params = {
            "latent_dim": latent_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": 50,
            "overwrite_existing_model": True
        }
        mlflow.log_params(trial_params)

        _, val_loss = training_nn(X_train_nn_short, X_val_nn_short, X_test_nn, trial_params, trial=trial) # Optuna params were injected
        mlflow.log_metric("val_loss", val_loss)
        return val_loss

def main():
    # Настраиваем MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("fraud_detection_pytorch_hpo")

    print("Starting tuning PyTorch hyperparameters with Optuna...")

    # Открываем "родительский" запуск в MLflow, чтобы сгруппировать все Trials
    # n_startup_trials - wait for first N trials for make statistics.
    # n_warmup_steps - wait for first N epochs for warming up.
    with mlflow.start_run(run_name="PyTorch_Optimization"):
        study = optuna.create_study(direction="minimize",
                                    study_name="PyTorch_Fraud_Tuning",
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))

        # 50 trials, n_jobs=-1 parallels execution.
        study.optimize(objective, n_trials=30, n_jobs=1)

        print("\n--- HPO finished ---")
        print(f"Best Val Loss (MSE): {study.best_value:.6f}")
        print("Best parameters:")

        best_params = study.best_params
        print(json.dumps(best_params, indent=4))

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_loss", study.best_value)


if __name__ == "__main__":
    main()