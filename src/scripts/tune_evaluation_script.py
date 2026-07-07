import json
import logging
import sys

import lightgbm as lgb
import pandas as pd

from src.paths import INFERENCE_PATH, LGBM_MODEL_PATH, CACHE_DIR
from src.pipelines.training_pipeline import evaluation_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)


def main():
    logging.info("Starting STANDALONE Evaluation mode.")

    if not LGBM_MODEL_PATH.exists():
        logging.error("LGBM model not found! Please run the full training pipeline first.")
        sys.exit(1)

    try:
        inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
        run_id = inference_meta["run_id"]
    except (FileNotFoundError, KeyError):
        logging.error("run_id not found in inference meta! Cannot log to MLflow.")
        sys.exit(1)

    logging.info("Loading trained LightGBM model from disk...")
    model_lgbm = lgb.Booster(model_file=str(LGBM_MODEL_PATH))

    logging.info("Loading cached parquet datasets...")
    train_df = pd.read_parquet(CACHE_DIR / "train_enriched.parquet")
    val_df = pd.read_parquet(CACHE_DIR / "val_enriched.parquet")
    test_df = pd.read_parquet(CACHE_DIR / "test_enriched.parquet")

    X_train, y_train = train_df.drop(columns=['isFraud']), train_df['isFraud']
    X_val, y_val = val_df.drop(columns=['isFraud']), val_df['isFraud']
    X_test, y_test = test_df.drop(columns=['isFraud']), test_df['isFraud']

    logging.info("Passing data to the evaluation pipeline...")
    evaluation_pipeline(model_lgbm, X_train, y_train, X_val, y_val, X_test, y_test, run_id, shap=False)

    logging.info("Standalone Evaluation finished successfully!")


if __name__ == "__main__":
    main()
