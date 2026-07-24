import logging

import pandas as pd


def inference_pipeline(data, inference_meta, model_lgbm):
    original_features = inference_meta["pytorch_features"]["original_features"]
    best_threshold = inference_meta["best_threshold"]
    lgbm_str_cols = inference_meta["pytorch_features"]["all_str_cols"]

    df_new_lgbm = pd.DataFrame([data])

    # ----------- LGBM Side ----------
    df_new_lgmb = df_new_lgbm.reindex(columns=original_features, fill_value=0)

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
