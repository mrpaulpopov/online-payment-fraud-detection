from sqlalchemy import text


async def fetch_lifetime_stats(db_connection):
    query = """
        SELECT uid1, 
        MAX("TransactionDT") AS last_tx_time,
        COUNT("TransactionID") AS tx_count,
        SUM("TransactionAmt") AS sum_amt,
        SUM("TransactionAmt" * "TransactionAmt") AS sum_amt_sq
        FROM test_final_features
        GROUP BY uid1;
        """

    result = await db_connection.execute(text(query))
    return result.mappings().all()


# 1 option: calculate -7 days from NOW
# 2 option: get the maximum timestamp from the table and calculate -7 days from it

async def fetch_7d_transactions(db_connection):
    query = """
        SELECT uid1, "TransactionID", "TransactionAmt", "DeviceType", "DeviceInfo", "TransactionDT"
        FROM test_final_features
        WHERE "TransactionDT" >= ((SELECT MAX("TransactionDT") FROM test_final_features) - 604800);
        """

    result = await db_connection.execute(text(query))
    return result.mappings().all()