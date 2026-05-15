import pandas as pd
import logging

def train_split(X, y, train_size=0.7, val_size=0.15):
    n = len(X)

    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    # fraud drift check:
    print(y_train.mean())
    print(y_val.mean())
    print(y_test.mean())
    logging.info('Splitting data completed.')
    return X_train, y_train, X_val, y_val, X_test, y_test