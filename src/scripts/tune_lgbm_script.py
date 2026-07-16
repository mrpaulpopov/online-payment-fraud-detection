import gc
import logging
import sys

import optuna
from optuna_integration import LightGBMPruningCallback
import lightgbm as lgb
import mlflow
import json
import yaml
from sklearn.metrics import average_precision_score

from src.data.data_loader import load_data
from src.data.split import train_split
from src.paths import CONFIG_PATH
from src.training.nn_data_preparation import pytorch_preprocessing, pytorch_filtering_rows, assign_anomaly_scores
from src.training.train_lgbm import prepare_data_for_lgbm
from src.training.train_nn import training_nn, pytorch_anomaly_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)

# Loading data
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
use_ae = config["pipeline"]["use_autoencoder"]
X, y, _ = load_data(table_name='train_final_features')
X_train, y_train, X_val, y_val, X_test, y_test = train_split(X, y, config["train_split"])

# ===========================================
# -------------- PyTorch side ---------------
# ===========================================
if use_ae:
    X_train_nn, X_val_nn, X_test_nn = pytorch_preprocessing(X_train, X_val, X_test, y_train, config["preprocessing"])
    X_train_nn_short, X_val_nn_short = pytorch_filtering_rows(X_train_nn, X_val_nn, y_train, y_val)

    logging.info('Starting PyTorch training')
    model_autoencoder, pt_val_loss = training_nn(X_train_nn_short, X_val_nn, X_test_nn, config["pytorch_params"])
    pt_params = {f"pt_{k}": v for k, v in config["pytorch_params"].items()}

    train_scores = pytorch_anomaly_scores(model_autoencoder, X_train_nn)
    val_scores = pytorch_anomaly_scores(model_autoencoder, X_val_nn)
    test_scores = pytorch_anomaly_scores(model_autoencoder, X_test_nn)
    X_train, X_val, X_test = assign_anomaly_scores(X_train, X_val, X_test, train_scores, val_scores, test_scores)

    del X_train_nn_short, X_train_nn, X_val_nn, X_test_nn, train_scores, val_scores, test_scores
    gc.collect()

# ===========================================
# -------------- LightGBM side --------------
# ===========================================
train_data, val_data, _, _, _, _ = prepare_data_for_lgbm(X_train, X_val, X_test, y_train, y_val, y_test)

def objective(trial):
    with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
        print("Trial started.")
        # Dependency between num_leaves and max_depth
        max_depth = trial.suggest_int("max_depth", 3, 6)
        max_possible_leaves = min(2 ** max_depth, 64) # num_leaves < (2^max_depth OR 150), 150 as fallback
        num_leaves = trial.suggest_int("num_leaves", 8, max_possible_leaves) # 2^3=8

        trial_params = {
            "objective": "binary",
            # "metric": "auc",              # maximizing ROC-AUC. But do we really need this?
            "metric": "average_precision",  # maximizing PR-AUC.
            "boosting_type": "gbdt",
            "random_state": 42,
            "verbose": -1,
            "feature_pre_filter": False,

            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True), # logarithmic changes
            "num_leaves": num_leaves,
            "max_depth": max_depth,

            # Regularization
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.85), # Take only 50-85% of rows.
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),                # Very powerful anti-overfit tool

            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.85),  # Take only 50-85% of columns.
            # "I found 100-1000 transaction with fraud. It is definitely a pattern."
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000),    # 1 pattern includes strictly 100-1000 tx
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.01, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 20.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 20.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
        }

        # Логируем параметры текущего эксперимента
        mlflow.log_params(trial_params)

        # Обучаем модель с ранней остановкой
        model = lgb.train(
            trial_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=3000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                LightGBMPruningCallback(trial, "average_precision") # average_precision - metric for pruning
            ]
        )

        # Оцениваем (Для задач фрода PR-AUC — идеальная метрика для максимизации)
        y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)
        pr_auc = average_precision_score(y_val, y_val_prob)

        # Логируем результат
        mlflow.log_metric("val_pr_auc", pr_auc)
        mlflow.log_metric("best_iteration", model.best_iteration)

    return pr_auc


def main():
    # Настраиваем MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("fraud_detection_lgbm_hpo")

    print("Starting tuning LGBM hyperparameters with Optuna...")

    # Открываем "родительский" запуск в MLflow, чтобы сгруппировать все Trials
    mlflow.end_run()
    with mlflow.start_run(run_name="LGBM_Optimization"):
        study = optuna.create_study(direction="maximize",
                                    study_name="LGBM_Fraud_Tuning",
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=7))

        # 50 trials, n_jobs=-1 parallels execution.
        study.optimize(objective, n_trials=170, n_jobs=1)

        print("\n--- HPO finished ---")
        print(f"Best PR-AUC: {study.best_value:.4f}")
        print("Best parameters:")

        best_params = study.best_params
        print(json.dumps(best_params, indent=4))

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_pr_auc", study.best_value)


if __name__ == "__main__":
    main()
