import pandas as pd

def main(file="/Users/paul/PycharmProjects/online-payment-fraud-detection/data/train_transaction.csv"):
    df = pd.read_csv(file, nrows=1000)  # берём часть для определения типов

    mapping = {
        "object": "TEXT",
        "int64": "BIGINT",
        "float64": "DOUBLE PRECISION",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP"
    }

    columns = []
    for col, dtype in df.dtypes.items():
        sql_type = mapping.get(str(dtype), "TEXT")
        columns.append(f'"{col}" {sql_type}')

    create_table = f"CREATE TABLE my_table (\n  " + ",\n  ".join(columns) + "\n);"
    print(create_table)

if __name__ == "__main__":
    main()