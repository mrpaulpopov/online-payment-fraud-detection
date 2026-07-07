import sys

from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import os
import time
from dotenv import load_dotenv
import logging


def load_data(table_name):
    start = time.time()
    load_dotenv() #.env

    try:
        engine = create_engine(f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
                                   f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}")
        engine.connect()
    except Exception as e:
        logging.error("PostgreSQL connection failed")
        # logging.error(f"PostgreSQL connection failed: {e.orig}") # Debug
        sys.exit(1)

    # Zero possibility of SQL-injections, therefore it's safe to use.
    query = f"""
    SELECT *
    FROM {table_name}
    """

    df_iter = pd.read_sql(query, engine, chunksize=50000)
    df = pd.concat(df_iter)
    # df = pd.read_sql(query, engine)

    df = df.sort_values("TransactionDT")
    df = df.reset_index(drop=True) # советуют после сортировки

    if "isFraud" in df.columns:
        y = df["isFraud"]
    else:
        y = None

    if "TransactionID" in df.columns:
        transaction_ids = df["TransactionID"]
    else:
        transaction_ids = None

    # 2. Drop useless columns
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT", #
                         "id_24", "id_25", "id_07", "id_08", "id_21", "id_26", "id_27", # trash values
                         "id_22", "id_23", "dist2", "D7", "id_18",                      # trash values
                         "uid1", "card1"], errors='ignore')                             # overfitting

    # Downsampling
    float64_cols = X.select_dtypes(include=['float64']).columns
    X[float64_cols] = X[float64_cols].astype('float32')
    logging.info(f"{table_name}: loading completed in {time.time() - start:.4f}s")
    # print(X.isnull().mean().sort_values(ascending=False)) # Analyze almost useless columns
    return X, y, transaction_ids