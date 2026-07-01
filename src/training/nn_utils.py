import torch
from torch.utils.data import TensorDataset, DataLoader


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def build_dataloader(X, batch_size, inference=False):
    '''
    This function can be used for both train and val data.
    Also it can be used in the inference (shuffle should be False during the inference).
    '''
    X_t = torch.tensor(X.to_numpy(), dtype=torch.float32)
    if not inference:
        dataset = TensorDataset(X_t, X_t)   # AutoEncoding
    else:
        dataset = TensorDataset(X_t)                # Inference
    return DataLoader(dataset, batch_size=batch_size, shuffle=(not inference))
