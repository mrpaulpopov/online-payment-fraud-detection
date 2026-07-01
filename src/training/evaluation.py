import json
import os

import mlflow
import seaborn as sns
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

    client.log_metric(run_id,f"lgbm_{prefix}_accuracy", accuracy_score(y, y_pred))  # useless
    client.log_metric(run_id,f"lgbm_{prefix}_precision", precision_score(y, y_pred))
    client.log_metric(run_id,f"lgbm_{prefix}_recall", recall_score(y, y_pred))
    client.log_metric(run_id,f"lgbm_{prefix}_f1", f1_score(y, y_pred))  # important
    client.log_metric(run_id,f"lgbm_{prefix}_roc_auc", roc_auc_score(y, y_prob))  # important
    client.log_metric(run_id,f"lgbm_{prefix}_pr_auc", average_precision_score(y, y_prob))  # important

    fpr, tpr, _ = roc_curve(y, y_prob)
    recall_at_fpr = max(tpr[fpr <= target_fpr])
    client.log_metric(run_id,f"lgbm_{prefix}_recall_at_fpr", recall_at_fpr)


def cross_validation(X_train, X_val, y_train, y_val, lgbm_params, n_splits):
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
            lgbm_params,
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

    mlflow.log_metric("lgbm_cv_pr_auc",
                      np.mean(scores))  # CV result averaged over 5 folds; provides a credible performance estimate.
    logging.info(f"CV completed in {time.time() - start:.4f}s")


def plot_shap_values(model, X_val, run_id):
    logging.info('Starting plot_shap_values')
    client = mlflow.MlflowClient()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val, show=False)
    save_path = PLOTS_DIR / 'shap_values.png'
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    client.log_artifact(run_id, save_path, 'plots')
    logging.info("Shap summary plots saved.")


def find_best_threshold(model, X_val, y_val, business_target_precision, threshold_strategy, run_id):
    '''
    Это автоматический способ нахождения threshold через business target (желаемый BUSINESS_TARGET_PRECISION)
    Threshold управляет переводом из probability 0.0-1.0 в decision 0-1 (not fraud, legit / fraud, to block).
    С какого момента probability считается fraud?
    '''
    logging.info('Starting find_best_threshold')

    client = mlflow.MlflowClient()
    y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)  # probability from 0.0 to 1.0
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)  # 'меню' всех возможных вариантов

    pr_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precisions[:-1],  # all but scikit last element
        'recall': recalls[:-1]  # all but scikit last element
    })

    # ========================================
    # ----------- BUSINESS TARGET ------------
    # ========================================

    # Оставляем только те строки, где Precision >= моего заданного значения
    good_precisions = pr_df[pr_df['precision'] >= business_target_precision]

    if not good_precisions.empty:
        # Сортируем по Recall по убыванию и берем самую первую строку (где Recall максимальный)
        best_row = good_precisions.sort_values(by='recall', ascending=False).iloc[0]
        best_business_threshold = best_row['threshold']
        logging.info(f"Business Target Precision {business_target_precision} achieved at threshold: {best_business_threshold}")
    else:
        best_business_threshold = 0.5  # fallback
        logging.warning("Business Target Precision is unreachable. Using default threshold 0.5.")

    client.log_param(run_id, "best_business_threshold", best_business_threshold)

    # ========================================
    # ----------- BEST F1-TARGET -------------
    # ========================================
    pr_df['f1_score'] = 2 * pr_df['precision'] * pr_df['recall'] / (pr_df['precision'] + pr_df['recall'] + 1e-08) # f1 formula
    best_row_index = pr_df['f1_score'].idxmax()
    best_row = pr_df.loc[best_row_index]

    best_f1_threshold = best_row['threshold']
    best_f1 = best_row['f1_score']
    best_precision = best_row['precision']
    best_recall = best_row['recall']
    logging.info(f"Max F1-Score {best_f1} achieved at threshold {best_f1_threshold}")
    logging.info(f"(Precision: {best_precision}, Recall: {best_recall})")
    client.log_param(run_id, "best_f1_threshold", best_f1_threshold)

    # ========================================
    # --------------- PLOTS ------------------
    # ========================================

    plt.figure(figsize=(10, 6))
    # Legit transactions
    sns.histplot(y_val_prob[y_val == 0], color='green', label='Legit (0)', stat="density", bins=50, alpha=0.5, kde=True)
    # Fraud transactions
    sns.histplot(y_val_prob[y_val == 1], color='red', label='Fraud (1)', stat="density", bins=50, alpha=0.5, kde=True)

    plt.axvline(best_business_threshold, color='orange', linestyle='--', label='Best Business threshold')
    plt.axvline(best_f1_threshold, color='blue', linestyle='--', label='Best F1 threshold')

    plt.title("Predicted Probability Distribution")
    plt.xlabel("Predicted Probability of Fraud")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()

    # Save to file
    save_path = PLOTS_DIR / 'probability_distribution.png'
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    # Save to MLflow
    client.log_artifact(run_id, save_path, 'plots')

    # ========================================
    # -------------- STRATEGY ----------------
    # ========================================

    if threshold_strategy == 'f1':
        final_threshold = best_f1_threshold
        logging.info(f"Strategy is 'f1'. Using F1 optimized threshold {final_threshold}.")
    elif threshold_strategy == 'business':
        final_threshold = best_business_threshold
        logging.info(f"Strategy is 'business'. Using Business Precision threshold {final_threshold}.")
    else:
        raise ValueError(f"Unknown threshold_strategy: {threshold_strategy}")

    # ========================================
    # ---------------- JSON ------------------
    # ========================================

    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    inference_meta["best_threshold"] = float(final_threshold)
    INFERENCE_PATH.write_text(json.dumps(inference_meta, indent=4), encoding="utf-8")

    return final_threshold

