import gc
import json
import logging
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.paths import IMPUTER_SCALER_PATH, INFERENCE_PATH


def pytorch_preprocessing(X_train, X_val, X_test, config):
    '''
    High cardinality filtering.
    '''
    logging.info('Starting PyTorch preprocessing')
    high_cardinality_threshold = config["high_cardinality_threshold"]

    # Train columns as a gold standard
    num_cols = X_train.select_dtypes(include=['number']).columns
    all_str_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns

    # --- ФИЛЬТР КАРДИНАЛЬНОСТИ (СПАСАЕТ RAM) ---
    str_cols = []
    for col in all_str_cols:
        # Оставляем только те колонки, где меньше N уникальных значений
        if X_train[col].nunique() < high_cardinality_threshold:
            str_cols.append(col)

    logging.info(f"Dropped high cardinality cols: {set(all_str_cols) - set(str_cols)}")

    # STRING COLUMNS: OHE
    str_train_data = X_train[str_cols].astype('string').fillna('missing')
    str_train_df = pd.get_dummies(str_train_data, dummy_na=False, dtype='float32')
    del str_train_data
    gc.collect()

    str_val_data = X_val[str_cols].astype('string').fillna('missing')
    str_val_df = pd.get_dummies(str_val_data, dummy_na=False, dtype='float32')
    del str_val_data
    gc.collect()

    str_test_data = X_test[str_cols].astype('string').fillna('missing')
    str_test_df = pd.get_dummies(str_test_data, dummy_na=False, dtype='float32')
    del str_test_data
    gc.collect()

    # In case of different val and test columns...
    str_val_df = str_val_df.reindex(columns=str_train_df.columns, fill_value=0).astype('float32')
    str_test_df = str_test_df.reindex(columns=str_train_df.columns, fill_value=0).astype('float32')

    # NUMERIC COLUMNS: Z-SCORE
    num_imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    num_train_data = num_imputer.fit_transform(X_train[num_cols])
    num_train_data = scaler.fit_transform(num_train_data).astype('float32')
    num_train_df = pd.DataFrame(num_train_data, columns=num_cols, index=X_train.index)

    del num_train_data
    gc.collect()

    # Saving for inference
    with IMPUTER_SCALER_PATH.open('wb') as f:
        pickle.dump({'imputer': num_imputer, 'scaler': scaler}, f)
    logging.info(f'Imputer and Scaler was saved to {IMPUTER_SCALER_PATH}')

    num_val_data = num_imputer.transform(X_val[num_cols])
    num_val_data = scaler.transform(num_val_data).astype('float32')
    num_val_df = pd.DataFrame(num_val_data, columns=num_cols, index=X_val.index)

    del num_val_data
    gc.collect()

    num_test_data = num_imputer.transform(X_test[num_cols])
    num_test_data = scaler.transform(num_test_data).astype('float32')
    num_test_df = pd.DataFrame(num_test_data, columns=num_cols, index=X_test.index)

    del num_test_data
    gc.collect()

    X_train_nn = pd.concat([str_train_df, num_train_df], axis=1)
    del str_train_df, num_train_df
    gc.collect()

    X_val_nn = pd.concat([str_val_df, num_val_df], axis=1)
    del str_val_df, num_val_df
    gc.collect()

    X_test_nn = pd.concat([str_test_df, num_test_df], axis=1)
    del str_test_df, num_test_df
    gc.collect()

    # Saving columns information for inference
    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    inference_meta["pytorch_features"] = {
        "num_cols": num_cols.tolist(),  # pandas indexes to list
        "str_cols": str_cols,  # is already list
        "all_str_cols": all_str_cols.tolist(),
        "original_features": X_train.columns.tolist(),
        "final_pytorch_features": X_train_nn.columns.tolist(),
    }
    INFERENCE_PATH.write_text(json.dumps(inference_meta, indent=4), encoding="utf-8")
    logging.info('PyTorch preprocessing finished and metadata saved')


    return X_train_nn, X_val_nn, X_test_nn


def pytorch_filtering_rows(X_train_nn, X_val_nn, y_train, y_val):
    '''
    For training the Autoencoder, we need only normal transactions (isFraud=0 rows).
    '''
    fraud0_mask_train = (y_train == 0).values
    X_train_nn_short = X_train_nn[fraud0_mask_train]
    fraud0_mask_val = (y_val == 0).values
    X_val_nn_short = X_val_nn[fraud0_mask_val]
    return X_train_nn_short, X_val_nn_short



def assign_anomaly_scores(X_train, X_val, X_test, train_scores, val_scores, test_scores):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_train['anomaly_score'] = train_scores
    X_val['anomaly_score'] = val_scores
    X_test['anomaly_score'] = test_scores
    return X_train, X_val, X_test
