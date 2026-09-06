import json
import logging
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from src.paths import INFERENCE_PATH, PLOTS_DIR

logger = logging.getLogger(__name__)


def evaluate_and_log_metrics(model: lgb.Booster, X: pd.DataFrame, y: pd.Series, best_threshold: float,
                             target_fpr: float, run_id: str, prefix: str):
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


def cross_validation(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, lgbm_params: dict, n_splits: int):
    # CV based on X_train+X_val sets. X_test should not be leaked.
    logger.info('Starting CV')
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
    logger.info(f"CV completed in {time.time() - start:.4f}s")


def plot_shap_values(model: lgb.Booster, X_val: pd.DataFrame, run_id: str):
    logger.info('Starting plot_shap_values')
    client = mlflow.MlflowClient()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val, show=False)
    save_path = PLOTS_DIR / 'shap_values.png'
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    client.log_artifact(run_id, save_path, 'plots')
    logger.info("Shap summary plots saved.")


def find_best_threshold(y_val: pd.Series, y_val_prob: np.ndarray, business_fp_target: float, threshold_strategy: str,
                        run_id: str) -> tuple[float, float, float]:
    '''
    The function returns three thresholds for plotting afterward.
    '''
    logger.info('Starting find_best_threshold')

    client = mlflow.MlflowClient()
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)  # 'menu' of all variants

    pr_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precisions[:-1],  # all but scikit last element
        'recall': recalls[:-1]  # all but scikit last element
    })

    # ========================================
    # ----------- BEST F1-TARGET -------------
    # ========================================

    pr_df['f1_score'] = 2 * pr_df['precision'] * pr_df['recall'] / (pr_df['precision'] + pr_df['recall'] + 1e-08) # f1 formula
    best_row_index = pr_df['f1_score'].idxmax()
    best_row = pr_df.loc[best_row_index]

    best_f1_threshold = best_row['threshold']
    logger.info(f"Max F1-Score achieved at threshold {best_f1_threshold}")
    client.log_param(run_id, "best_f1_threshold", best_f1_threshold)

    # ========================================
    # ----------- BUSINESS TARGET ------------
    # ========================================

    # Leave only the rows where Precision >= user target value
    business_target_precision = (1 - business_fp_target)
    good_precisions = pr_df[pr_df['precision'] >= business_target_precision]

    if not good_precisions.empty:
        # Sort Recall in descending order and get the first (maximum) row
        best_row = good_precisions.sort_values(by='recall', ascending=False).iloc[0]
        best_business_threshold = best_row['threshold']
        logger.info(
            f"Business Target Precision {business_target_precision} achieved at threshold: {best_business_threshold}")
    else:
        best_business_threshold = best_f1_threshold  # fallback
        logger.warning("Business Target Precision is unreachable. Using best f1 threshold.")

    client.log_param(run_id, "best_business_threshold", best_business_threshold)

    # ========================================
    # -------------- STRATEGY ----------------
    # ========================================

    if threshold_strategy == 'f1':
        final_threshold = best_f1_threshold
        logger.info(f"Strategy is 'f1'. Using F1 optimized threshold {final_threshold}.")
    elif threshold_strategy == 'business':
        final_threshold = best_business_threshold
        logger.info(f"Strategy is 'business'. Using Business Precision threshold {final_threshold}.")
    else:
        raise ValueError(f"Unknown threshold_strategy: {threshold_strategy}")

    # ========================================
    # ---------------- JSON ------------------
    # ========================================

    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    inference_meta["best_threshold"] = float(final_threshold)
    INFERENCE_PATH.write_text(json.dumps(inference_meta, indent=4), encoding="utf-8")

    return final_threshold, best_business_threshold, best_f1_threshold
