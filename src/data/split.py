import pandas as pd
import logging

def train_split(X, y, config):
    train_size = config['train_size']
    test_size = config['test_size']

    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be greater than zero")
    if train_size + test_size >= 1:
        raise ValueError("There's no data for validation, decrease train_size or test_size")

    n = len(X)
    if n == 0:
        raise ValueError("Dataset is empty")

    train_end = int(n * train_size)
    test_start = n - int(n * test_size)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:test_start]
    y_val = y.iloc[train_end:test_start]
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    logging.info(f"Fraud drift check after train-test split: Train {y_train.mean()}, Val: {y_val.mean()}, Test: {y_test.mean()}")
    logging.info('Splitting data completed.')
    return X_train, y_train, X_val, y_val, X_test, y_test