import gc
import pickle
import lightgbm as lgb
from sympy.stats.rv import probability

from src.data.data_loader import load_data
from src.data.split import train_split
from src.models.autoencoder import autoencoder_nn
from src.paths import IMPUTER_SCALER_PATH, LGBM_MODEL_PATH, INFERENCE_PATH
import json
import torch
import time
import pandas as pd
from src.inference_test import new_data
from src.paths import NN_MODEL_PATH
import torch.nn.functional as F
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def main():
    logging.info('Starting Kaggle-inference pipeline')
    start = time.time()

    # Read JSON
    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    best_threshold = inference_meta["best_threshold"]
    input_dim = inference_meta["input_dim"]
    latent_dim = inference_meta["latent_dim"]
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

    # --------------------------------------
    # --------- Reading Test table ---------
    # --------------------------------------

    X_test, _, kaggle_ids = load_data(table_name='test_final_features')

    df_nn = pd.DataFrame(X_test)
    df_lgmb = df_nn.copy()

    # --------------------------------
    # --------- PyTorch Side ---------
    # --------------------------------
    logging.info('Starting PyTorch side')

    model = autoencoder_nn(input_dim, latent_dim)
    model.load_state_dict(torch.load(NN_MODEL_PATH, weights_only=True))
    model.eval()

    # NUMERIC COLUMNS: Z-SCORE
    df_nn[num_cols] = num_imputer.transform(df_nn[num_cols]) # transforming nulls to mean
    df_nn[num_cols] = scaler.transform(df_nn[num_cols]) # z-score
    num_df = df_nn[num_cols].astype('float32')

    # STRING COLUMNS: OHE
    df_nn[str_cols] = df_nn[str_cols].astype('string').fillna('missing')
    str_df = pd.get_dummies(df_nn[str_cols], dummy_na=False, dtype='float32')

    df_nn = pd.concat([num_df, str_df], axis=1)
    df_nn = df_nn.reindex(columns=final_pytorch_features, fill_value=0).astype('float32')

    with torch.no_grad():
        tensor_data = torch.tensor(df_nn.values)
        prediction = model(tensor_data)
        anomaly_score = F.mse_loss(prediction, tensor_data).item()

    del df_nn, str_df, num_df
    gc.collect()


    # --------------------------------
    # ----------- LGBM Side ----------
    # --------------------------------
    logging.info('Starting LGBM side')

    model_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)
    df_lgmb = df_lgmb.reindex(columns=original_features, fill_value=0)
    df_lgmb['anomaly_score'] = anomaly_score

    for col in df_lgmb.columns:
        if col in lgbm_str_cols:
            df_lgmb[col] = df_lgmb[col].astype('str').astype('category')
        else:
            df_lgmb[col] = pd.to_numeric(df_lgmb[col], errors='coerce')

    # # Check columns
    # extra_columns = set(df_lgmb.columns) - set(model_lgbm.feature_name())
    # print(f"Checking extra columns: {extra_columns}")


    pred_proba = model_lgbm.predict(df_lgmb)
    pred_class = (pred_proba > best_threshold).astype(int)

    # --------------------------------
    # ----------- Export -------------
    # --------------------------------
    logging.info('Starting Export')

    submission = pd.DataFrame({
        'TransactionID': kaggle_ids,
        'isFraud': pred_class
    })
    submission.to_csv('my_submission.csv', index=False)

    logging.info(f"Kaggle inference completed in {time.time() - start:.4f}s")

if __name__ == '__main__':
    main()