import logging

import pandas as pd
import torch
import torch.nn.functional as F


def inference_pipeline(data, inference_meta, num_imputer, scaler, model_lgbm, model_pytorch):
    original_features = inference_meta["pytorch_features"]["original_features"]
    best_threshold = inference_meta["best_threshold"]
    final_pytorch_features = inference_meta["pytorch_features"]["final_pytorch_features"]
    num_cols = inference_meta["pytorch_features"]["num_cols"]
    str_cols = inference_meta["pytorch_features"]["str_cols"]
    lgbm_str_cols = inference_meta["pytorch_features"]["all_str_cols"]

    df_new_nn = pd.DataFrame([data])
    df_new_lgmb = df_new_nn.copy()

    # --------- PyTorch Side ---------
    # NUMERIC COLUMNS: Z-SCORE
    df_new_nn[num_cols] = num_imputer.transform(df_new_nn[num_cols])  # transforming nulls to mean
    df_new_nn[num_cols] = scaler.transform(df_new_nn[num_cols])  # z-score
    num_df = df_new_nn[num_cols].astype('float32')

    # STRING COLUMNS: OHE
    df_new_nn[str_cols] = df_new_nn[str_cols].astype('string').fillna('missing')
    str_df = pd.get_dummies(df_new_nn[str_cols], dummy_na=False, dtype='float32')

    df_new_nn = pd.concat([num_df, str_df], axis=1)
    df_new_nn = df_new_nn.reindex(columns=final_pytorch_features, fill_value=0).astype('float32')

    with torch.no_grad():
        tensor_data = torch.tensor(df_new_nn.values)
        prediction = model_pytorch(tensor_data)
        anomaly_score = F.mse_loss(prediction, tensor_data).item()

    # ----------- LGBM Side ----------
    df_new_lgmb = df_new_lgmb.reindex(columns=original_features, fill_value=0)
    df_new_lgmb['anomaly_score'] = anomaly_score

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
