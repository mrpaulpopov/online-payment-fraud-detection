import gc
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from src.paths import NN_MODEL_PATH
from src.models.autoencoder import autoencoder_nn
from src.paths import IMPUTER_SCALER_PATH


def pytorch_preprocessing(X_train, X_val, X_test, y_train, y_val, y_test, high_cardinality_threshold):
    logging.info('Starting PyTorch preprocessing')

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

    # STRINGS
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

    result_train = pd.concat([str_train_df, num_train_df], axis=1)
    del str_train_df, num_train_df
    gc.collect()

    result_val = pd.concat([str_val_df, num_val_df], axis=1)
    del str_val_df, num_val_df
    gc.collect()

    result_test = pd.concat([str_test_df, num_test_df], axis=1)
    del str_test_df, num_test_df
    gc.collect()

    return result_train, result_val, result_test


def build_dataloader(X_data, batch_size):  # function is used for both train and val data
    X_data_t = torch.tensor(X_data.to_numpy(), dtype=torch.float32)
    dataset = TensorDataset(X_data_t, X_data_t)  # AutoEncoder
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def train_nn_loop(model, train_loader, val_loader, test_loader, optimizer, loss_fn, N_EPOCHS):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    logging.info(f"Starting training on {DEVICE}")

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        for X_train_batch, target_batch in train_loader:
            X_train_batch = X_train_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            preds = model(X_train_batch)

            loss = loss_fn(preds, target_batch)  # FORWARD pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # sum of losses.
        train_loss = total_loss / len(train_loader)  # 'normalized' loss

        # ===== VALIDATION =====
        model.eval()
        val_loss_sum = 0
        val_sq_err_sum = 0
        val_abs_err_sum = 0
        total_val_elements = 0

        with torch.no_grad():
            for X_val_batch, target_batch in val_loader:
                X_val_batch = X_val_batch.to(DEVICE)
                target_batch = target_batch.to(DEVICE)

                preds = model(X_val_batch)
                loss = loss_fn(preds, target_batch)

                val_loss_sum += loss.item()

                # Считаем сумму ошибок на лету (сохраняем только одно число .item())
                val_sq_err_sum += torch.sum((preds - target_batch) ** 2).item()
                val_abs_err_sum += torch.sum(torch.abs(preds - target_batch)).item()
                total_val_elements += target_batch.numel()  # Общее количество чисел в батче

        val_loss = val_loss_sum / len(val_loader)
        rmse = np.sqrt(val_sq_err_sum / total_val_elements)
        mae = val_abs_err_sum / total_val_elements

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_rmse={rmse:.4f} | "
            f"val_mae={mae:.4f}"
        )

    gc.collect()
    logging.info('Starting evaluation on Test set')

    model.eval()
    test_loss_sum = 0
    test_sq_err_sum = 0
    test_abs_err_sum = 0
    total_test_elements = 0
    with torch.no_grad():
        for X_test_batch, target_batch in test_loader:
            X_test_batch = X_test_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            preds = model(X_test_batch)
            loss = loss_fn(preds, target_batch)

            test_loss_sum += loss.item()

            test_sq_err_sum += torch.sum((preds - target_batch) ** 2).item()
            test_abs_err_sum += torch.sum(torch.abs(preds - target_batch)).item()
            total_test_elements += target_batch.numel()

    if len(test_loader) > 0:
        test_loss = test_loss_sum / len(test_loader)
        test_rmse = np.sqrt(test_sq_err_sum / total_test_elements)
        test_mae = test_abs_err_sum / total_test_elements

        print("-" * 50)
        print(
            f"FINAL TEST METRICS | "
            f"test_loss={test_loss:.4f} | "
            f"test_rmse={test_rmse:.4f} | "
            f"test_mae={test_mae:.4f}"
        )
        print("-" * 50)

    torch.save(model.state_dict(), NN_MODEL_PATH)
    logging.info(f"Model saved to {NN_MODEL_PATH}")


def training_nn(X_train, X_val, X_test, LEARNING_RATE, BATCH_SIZE, N_EPOCHS):
    # Dimensions
    input_dim = X_train.shape[1]

    model = autoencoder_nn(input_dim)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_loader = build_dataloader(X_train, BATCH_SIZE)
    val_loader = build_dataloader(X_val, BATCH_SIZE)
    test_loader = build_dataloader(X_test, BATCH_SIZE)

    train_nn_loop(model, train_loader, val_loader, test_loader, optimizer, loss_fn, N_EPOCHS)
    return model


def get_anomaly_scores(model, X_processed, batch_size=1024):
    device = next(model.parameters()).device  # get first element of the iterator (parameters of the model)
    X_t = torch.tensor(X_processed.to_numpy(), dtype=torch.float32)

    # DataLoader
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # shuffle should be False during the inference

    model.eval()
    scores = []

    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch[0].to(device)

            preds = model(X_batch)

            row_losses = torch.mean((preds - X_batch) ** 2, dim=1)  # 'sum' of losses in the row. MSE
            scores.extend(row_losses.cpu().numpy())  # scores: X_batch+X_batch+X_batch+...

    return np.array(scores)


def assign_anomaly_scores(X_train, X_val, X_test, train_scores, val_scores, test_scores):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train['anomaly_score'] = train_scores
    X_val['anomaly_score'] = val_scores
    X_test['anomaly_score'] = test_scores
    return X_train, X_val, X_test