import pickle
import lightgbm as lgb

from src.models.autoencoder import autoencoder_nn
from src.paths import IMPUTER_SCALER_PATH, LGBM_MODEL_PATH, INFERENCE_PATH
import json
import torch
import pandas as pd
from src.inference_test import new_data
from src.paths import NN_MODEL_PATH
import torch.nn.functional as F

def inference_pipeline():
    # Read JSON
    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    best_threshold = inference_meta["best_threshold"]
    input_dim = inference_meta["input_dim"]
    original_features = inference_meta["pytorch_features"]["original_features"]
    final_pytorch_features = inference_meta["pytorch_features"]["final_pytorch_features"]
    num_cols = inference_meta["pytorch_features"]["num_cols"]
    str_cols = inference_meta["pytorch_features"]["str_cols"]
    lgbm_str_cols = inference_meta["pytorch_features"]["all_str_cols"]

    # Read Imputer, Scaler
    with IMPUTER_SCALER_PATH.open('rb') as f:
        imp_object= pickle.load(f)
    num_imputer = imp_object['imputer']
    scaler = imp_object['scaler']

    df_new_nn = pd.DataFrame([new_data])
    df_new_lgmb = df_new_nn.copy()

    # --------------------------------
    # --------- PyTorch Side ---------
    # --------------------------------
    model = autoencoder_nn(input_dim)
    model.load_state_dict(torch.load(NN_MODEL_PATH))
    model.eval()

    # NUMERIC COLUMNS: Z-SCORE
    df_new_nn[num_cols] = num_imputer.transform(df_new_nn[num_cols]) # transforming nulls to mean
    df_new_nn[num_cols] = scaler.transform(df_new_nn[num_cols]) # z-score
    num_df = df_new_nn[num_cols].astype('float32')

    # STRING COLUMNS: OHE
    df_new_nn[str_cols] = df_new_nn[str_cols].astype('string').fillna('missing')
    str_df = pd.get_dummies(df_new_nn[str_cols], dummy_na=False, dtype='float32')

    df_new_nn = pd.concat([num_df, str_df], axis=1)
    df_new_nn = df_new_nn.reindex(columns=final_pytorch_features, fill_value=0).astype('float32')

    with torch.no_grad():
        tensor_data = torch.tensor(df_new_nn.values)
        prediction = model(tensor_data)
        anomaly_score = F.mse_loss(prediction, tensor_data).item()

    # --------------------------------
    # ----------- LGBM Side ----------
    # --------------------------------

    model_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)
    df_new_lgmb = df_new_lgmb.reindex(columns=original_features, fill_value=0)
    df_new_lgmb['anomaly_score'] = anomaly_score


    for col in df_new_lgmb.columns:
        if col in lgbm_str_cols:
            df_new_lgmb[col] = df_new_lgmb[col].astype('str').astype('category')
        else:
            df_new_lgmb[col] = pd.to_numeric(df_new_lgmb[col], errors='coerce')


    predictions = model_lgbm.predict(df_new_lgmb)
    predicted_classes = (predictions > best_threshold).astype(int)
    print(predicted_classes)