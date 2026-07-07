import psycopg2
import sys

# Conversion PostgreSQL -> Python/Pydantic
PG_TO_PY = {
    "integer": "int",
    "bigint": "int",
    "smallint": "int",
    "serial": "int",
    "bigserial": "int",
    "numeric": "Decimal",
    "decimal": "Decimal",
    "real": "float",
    "double precision": "float",
    "money": "Decimal",
    "character varying": "str",
    "character": "str",
    "text": "str",
    "uuid": "UUID",
    "boolean": "bool",
    "date": "date",
    "timestamp without time zone": "datetime",
    "timestamp with time zone": "datetime",
    "time without time zone": "time",
    "time with time zone": "time",
    "json": "dict",
    "jsonb": "dict",
    "bytea": "bytes",
    "ARRAY": "list",
    "inet": "str",
    "cidr": "str",
}


def fetch_columns(conn, table, schema="public"):
    query = """
        SELECT column_name, data_type, udt_name, is_nullable, column_default,
               character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position;
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        rows = cur.fetchall()
    if not rows:
        sys.exit(f"Таблица {schema}.{table} не найдена или в ней нет столбцов.")
    return rows


def pg_type_to_py(data_type: str, udt_name: str) -> str:
    if data_type == "ARRAY":
        # udt_name для массива обычно вида "_int4", "_text" и т.п.
        base = udt_name.lstrip("_")
        base_map = {
            "int4": "int", "int8": "int", "int2": "int",
            "text": "str", "varchar": "str", "numeric": "Decimal",
            "bool": "bool", "float8": "float", "float4": "float",
            "uuid": "UUID", "timestamp": "datetime", "timestamptz": "datetime",
        }
        py_base = base_map.get(base, "str")
        return f"list[{py_base}]"
    return PG_TO_PY.get(data_type, "str")  # если тип неизвестен — str как безопасный дефолт


def build_model_code(table: str, columns, class_name) -> str:
    fields = []

    for column_name, data_type, udt_name, is_nullable, column_default, max_len in columns:
        py_type = pg_type_to_py(data_type, udt_name)

        if max_len:
            comment += f"({max_len})"

        annotation = f"{py_type} | None = None"

        fields.append(f"    {column_name}: {annotation}")

    code = f"class {class_name}(BaseModel):\n"
    code += "\n".join(fields)
    code += "\n"
    return code



def main(table, schema, host, port, dbname, user, password, class_name):
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password,
    )
    try:
        columns = fetch_columns(conn, table, schema)
        code = build_model_code(table, columns, class_name)
    finally:
        conn.close()

    print(code)

if __name__ == "__main__":
    main(table="train_final_features", schema="public", host="localhost", port=5432,
         dbname="bankdb", user="user", password="password", class_name="Transaction")