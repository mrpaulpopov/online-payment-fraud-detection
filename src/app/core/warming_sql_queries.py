from sqlalchemy import text

async def fetch_lifetime_stats(db_connection):
    query = f"""
        SELECT uid1, 
        MAX("TransactionDT") AS last_tx_time,
        COUNT("TransactionID") AS tx_count,
        SUM("TransactionAmt") AS sum_amt,
        SUM("TransactionAmt" * "TransactionAmt") AS sum_amt_sq
        FROM test_final_features
        GROUP BY uid1;
        """

    result = db_connection.execute(text(query))
    return result.mapping().all()


async def fetch_7d_transactions(db_connection):
    query = f"""
        SELECT uid1, "TransactionID", "TransactionAmt", "DeviceType", "DeviceInfo", "TransactionDT"
        FROM test_final_features
        WHERE "TransactionDT" >= (EXTRACT(EPOCH FROM NOW()) - 604800);
        """

    result = db_connection.execute(text(query))
    db_connection.execute(text(query))
    return result.mapping().all()