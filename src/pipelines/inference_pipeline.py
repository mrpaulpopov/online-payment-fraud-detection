import logging

import pandas as pd


def inference_pipeline(data, inference_meta, model_lgbm):
    logging.info("Starting Inference Pipeline")
    original_features = inference_meta["features"]["original_features"]
    best_threshold = inference_meta["best_threshold"]
    lgbm_str_cols = inference_meta["features"]["all_str_cols"]

    df_lgbm = pd.DataFrame([data])
    df_new_lgmb = df_lgbm.reindex(columns=original_features, fill_value=0)

    for col in df_new_lgmb.columns:
        if col in lgbm_str_cols:
            df_new_lgmb[col] = df_new_lgmb[col].astype('str').astype('category')
        else:
            df_new_lgmb[col] = pd.to_numeric(df_new_lgmb[col], errors='coerce')


    pred_proba = model_lgbm.predict(df_new_lgmb)
    pred_class = (pred_proba > best_threshold).astype(int)
    logging.info(f"Probability: {pred_proba}")
    logging.info(f"Predicted class: {pred_class}")

    return pred_proba, pred_class
