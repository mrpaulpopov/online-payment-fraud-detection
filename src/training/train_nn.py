import gc
import json
import logging
import copy

import numpy as np
import optuna
import torch
from optuna.testing import trials
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from src.paths import NN_MODEL_PATH, INFERENCE_PATH
from src.models.autoencoder import autoencoder_nn
from src.training.nn_utils import EarlyStopping, build_dataloader


def train_nn_loop(model, train_loader, val_loader, test_loader, optimizer, loss_fn, N_EPOCHS, pytorch_params, trial=None):
    '''
    Child function of training_nn.
    '''
    overwrite_existing_model = pytorch_params["overwrite_existing_model"]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    logging.info(f"Starting training on {DEVICE}")

    val_loss = None # protection for return
    early_stopping = EarlyStopping(patience=5)
    best_model_weights = copy.deepcopy(model.state_dict())

    if not NN_MODEL_PATH.exists() or overwrite_existing_model:
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

            logging.info(
                f"Epoch {epoch + 1}/{N_EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_rmse={rmse:.4f} | "
                f"val_mae={mae:.4f}"
            )
            # =================================
            # --------- OPTUNA PRUNING --------
            # =================================
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    logging.info(f"Trial pruned at epoch {epoch+1}!")
                    raise optuna.TrialPruned()

            # =================================
            # --------- EARLY STOPPING --------
            # =================================
            if early_stopping.best_loss is None or val_loss < early_stopping.best_loss:
                best_model_weights = copy.deepcopy(model.state_dict()) # copy weights only if new val_loss is lower
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info(f'Early stopping. Stop training')
                break

        gc.collect()
        model.load_state_dict(best_model_weights)
        val_loss = early_stopping.best_loss

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
    else:
        logging.info(f"Loading existing model from {NN_MODEL_PATH}")
        model.load_state_dict(torch.load(NN_MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
        val_loss_sum = 0

        with torch.no_grad():
            for X_val_batch, target_batch in val_loader:
                X_val_batch = X_val_batch.to(DEVICE)
                target_batch = target_batch.to(DEVICE)

                preds = model(X_val_batch)
                loss = loss_fn(preds, target_batch)
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)
        logging.info(f"Loaded model val_loss={val_loss:.4f}")
    return val_loss


def training_nn(X_train, X_val, X_test, pytorch_params, trial=None):
    '''
    Entry point for PyTorch training.
    '''
    # Dimensions
    input_dim = X_train.shape[1]

    # Reading Config
    latent_dim = pytorch_params["latent_dim"]
    learning_rate = pytorch_params["learning_rate"]
    batch_size = pytorch_params["batch_size"]
    n_epochs = pytorch_params["n_epochs"]

    # Saving input_dim and latent_dim
    # "Append" JSON: Read-Append-Write
    inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    inference_meta["input_dim"] = int(input_dim)
    inference_meta["latent_dim"] = int(latent_dim)
    INFERENCE_PATH.write_text(json.dumps(inference_meta, indent=4), encoding="utf-8")

    model = autoencoder_nn(input_dim, latent_dim)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_loader = build_dataloader(X_train, batch_size)
    val_loader = build_dataloader(X_val, batch_size)
    test_loader = build_dataloader(X_test, batch_size)

    val_loss = train_nn_loop(model, train_loader, val_loader, test_loader, optimizer, loss_fn, n_epochs, pytorch_params, trial=trial)
    return model, val_loss


def pytorch_anomaly_scores(model, X_processed, batch_size=1024):
    '''
    Inference function.
    '''
    device = next(model.parameters()).device  # get first element of the iterator (parameters of the model)

    loader = build_dataloader(X_processed, batch_size, inference=True)

    model.eval()
    scores = []

    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch[0].to(device)

            preds = model(X_batch)

            row_losses = torch.mean((preds - X_batch) ** 2, dim=1)  # 'sum' of losses in the row. MSE
            scores.extend(row_losses.cpu().numpy())  # scores: X_batch+X_batch+X_batch+...

    return np.array(scores)


