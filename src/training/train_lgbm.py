from sklearn.model_selection import StratifiedKFold
from src.data.split import train_split
from src.data.data_loader import load_data
import lightgbm as lgb
import pandas as pd
import shap
import numpy as np
import logging
import mlflow
import time
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve
)

# TODO: move to main.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def data_preparation(X, y):
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        X[col] = X[col].astype("category")  # Convert 'string' columns to 'category' (LightGBM requirement)

    X_train, y_train, X_val, y_val, X_test, y_test = train_split(X, y, train_size=0.7, val_size=0.15)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Manual calculation of scale_pos_weight
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / num_pos
    # print(scale_pos_weight) # (27)

    return X_train, y_train, X_val, y_val, X_test, y_test, train_data, valid_data


def training_lgbm(train_data, valid_data, params):
    start = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,  # 1000-3000
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50)]
    )

    mlflow.log_params(params)  # logging hyperparameters
    mlflow.lightgbm.log_model(model, name="model")
    mlflow.log_metric("best_iteration", model.best_iteration)
    print(f"Training completed in {time.time() - start:.4f}s")

    return model


def find_best_threshold(model, X_val, y_val, target_precision):
    # Threshold управляет переводом из probability 0.0-1.0 в decision 0-1 (not fraud, legit / fraud, to block).
    # С какого момента probability считается fraud?

    # Это автоматический способ нахождения threshold через business target (желаемый TARGET_PRECISION)
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

    mlflow.log_param("best_threshold", best_threshold)
    mlflow.log_param("target_precision", target_precision)  # customized parameter

    return best_threshold


def evaluate_and_log_metrics(model, X, y, best_threshold, target_fpr, prefix):
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
    mlflow.log_metric(f"{prefix}_recall_at_fpr", recall_at_fpr)


def cross_validation(X_train, X_val, y_train, y_val, params):
    # CV based on X_train+X_val sets. X_test should not be leaked.
    X_cv = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_cv = pd.concat([y_train, y_val]).reset_index(drop=True)

    start = time.time()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
        preds = cv_model.predict(X_val_fold)
        scores.append(average_precision_score(y_val_fold, preds))

    mlflow.log_metric("cv_pr_auc",
                      np.mean(scores))  # CV result averaged over 5 folds; provides a credible performance estimate.
    print(f"CV completed in {time.time() - start:.4f}s")


def plot_shap_values(model, X_val):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val)
    shap.summary_plot(shap_values, X_val)


def false_positives_handling(model, X_val, y_val, best_threshold):
    y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)
    y_val_pred = (y_val_prob > best_threshold).astype(int)

    # Создаем маску для ложных срабатываний (в реальности фрода нет, но модель сказала "да")
    fp_mask = (y_val == 0) & (y_val_pred == 1)
    false_positives = X_val[fp_mask].copy()

    # Добавляем предсказанные вероятности, чтобы видеть уверенность модели
    false_positives['predict_prob'] = y_val_prob[fp_mask]
    print(f"Количество ложных срабатываний: {len(false_positives)}")
    # Выводим топ-10 самых "уверенных" ошибок модели
    print(false_positives.sort_values(by='predict_prob', ascending=False).head(10))


def training_lgbm_pipeline():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    params = config["lgbm_params"]
    target_precision = config["business_targets"]["target_precision"]
    target_fpr = config["business_targets"]["target_fpr"]

    X, y = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, train_data, valid_data = data_preparation(X, y)
    mlflow.set_experiment("fraud_detection")
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run():
        model = training_lgbm(train_data, valid_data, params)
        best_threshold = find_best_threshold(model, X_val, y_val, target_precision)
        evaluate_and_log_metrics(model, X_train, y_train, best_threshold, target_fpr, prefix='train')
        evaluate_and_log_metrics(model, X_val, y_val, best_threshold, target_fpr, prefix='val')
        cross_validation(X_train, X_val, y_train, y_val, params)

    plot_shap_values(model, X_val)
    false_positives_handling(model, X_val, y_val, best_threshold)


training_lgbm_pipeline()

# mlflow ui
# http://127.0.0.1:5000
