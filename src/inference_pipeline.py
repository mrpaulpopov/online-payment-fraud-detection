import pickle
import lightgbm as lgb
from src.paths import IMPUTER_SCALER_PATH, LGBM_MODEL_PATH, INFERENCE_PATH
import json
import pandas as pd
from src.inference_test import new_data

inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
best_threshold = inference_meta["best_threshold"]

with IMPUTER_SCALER_PATH.open('rb') as f:
    imp_object= pickle.load(f)
num_imputer = imp_object['imputer']
scaler = imp_object['scaler']

model_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)

df_newdata = pd.DataFrame([new_data])
num_cols = df_newdata.select_dtypes(include=['number']).columns



df_newdata[num_cols] = num_imputer.transform(df_newdata[num_cols])
df_newdata[num_cols] = scaler.transform(df_newdata[num_cols])

df_newdata_normalized


predictions = model_lgbm.predict(new_data)
predicted_classes = (predictions > best_threshold).astype(int)