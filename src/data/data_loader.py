from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import os
import time
from dotenv import load_dotenv


def load_data():
    start = time.time()
    load_dotenv() #.env
    engine = create_engine(f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
                                   f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}")
    engine.connect()

    query = """
    SELECT *
    FROM final_features
    """

    df_iter = pd.read_sql(query, engine, chunksize=50000)
    df = pd.concat(df_iter)
    # df = pd.read_sql(query, engine)

    # 2. Drop useless columns
    X = df.drop(columns=["isFraud", "TransactionID", "id_24", "id_25", "id_07", "id_08", "id_21", "id_26", "id_27", "id_22", "id_23", "dist2", "D7", "id_18"])
    y = df["isFraud"]

    print(f"Database loading completed in {time.time() - start:.4f}s")
    print(X.shape)
    print(X.isnull().mean().sort_values(ascending=False)) # 1. Analyze almost useless columns
    return X, y

load_data()