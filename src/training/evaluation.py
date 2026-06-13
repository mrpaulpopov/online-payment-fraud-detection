import json
import os

import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score, precision_recall_curve,
)
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np
import shap
import logging
import matplotlib.pyplot as plt
from src.paths import PLOTS_DIR, INFERENCE_PATH


def evaluate_and_log_metrics(model, X, y, best_threshold, target_fpr, run_id, prefix=None):
    client = mlflow.MlflowClient()
    y = y.astype(int)
    y_prob = model.predict(X, num_iteration=model.best_iteration)  # probability from 0.0 to 1.0
    y_pred = (y_prob > best_threshold).astype(int)  # astype(int) converts False/True to 0/1

    mlflow.log_metric(f"{prefix}_accuracy", accuracy_score(y, y_pred))  # useless
    mlflow.log_metric(f"{prefix}_precision", precision_score(y, y_pred))
    mlflow.log_metric(f"{prefix}_recall", recall_score(y, y_pred))
    mlflow.log_metric(f"{prefix}_f1", f1_score(y, y_pred))  # important
    mlflow.log_metric(f"{prefix}_roc_auc", roc_auc_score(y, y_prob))  # important
    mlflow.log_metric(f"{prefix}_pr_auc", average_precision_score(y, y_prob))  # important

    fpr, tpr, _ = roc_curve(y, y_prob)
    recall_at_fpr = max(tpr[fpr <= target_fpr])
    client.log_metric(run_id,f"{prefix}_recall_at_fpr", recall_at_fpr)


def cross_validation(X_train, X_val, y_train, y_val, params, n_splits):
    # CV based on X_train+X_val sets. X_test should not be leaked.
    logging.info('Starting CV')
    X_cv = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_cv = pd.concat([y_train, y_val]).reset_index(drop=True)

    categorical_cols = X_cv.select_dtypes(include=["object", "string"]).columns
    X_cv[categorical_cols] = X_cv[categorical_cols].astype("category")

    start = time.time()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_cv, y_cv):
        X_train_fold, X_val_fold = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_train_fold, y_val_fold = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

        train_cv_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_cv_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_cv_set)

        cv_model = lgb.train(  # Temporary 'cv_model' to avoid overwriting the main trained 'model'
            params,
            train_cv_set,
            num_boost_round=500,
            valid_sets=[val_cv_set],
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(0)
            ]
        )
        preds = cv_model.predict(X_val_fold, num_iteration=cv_model.best_iteration)
        scores.append(average_precision_score(y_val_fold, preds))

    mlflow.log_metric("cv_pr_auc",
                      np.mean(scores))  # CV result averaged over 5 folds; provides a credible performance estimate.
    logging.info(f"CV completed in {time.time() - start:.4f}s")


def plot_shap_values(model, X_val, run_id):
    client = mlflow.MlflowClient()
    logging.info('Starting plot_shap_values')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val, show=False)
    save_path = PLOTS_DIR / 'shap_values.png'
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    client.log_artifact(run_id, save_path, 'plots')
    logging.info("Shap summary plots saved.")


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

    # Saving best_threshold
    # "Append" JSON: Read-Append-Write
    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    inference_meta["best_threshold"] = float(best_threshold)
    INFERENCE_PATH.write_text(json.dumps(inference_meta, indent=4), encoding="utf-8")

    return best_threshold
