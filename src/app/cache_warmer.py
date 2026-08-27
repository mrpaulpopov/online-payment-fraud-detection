import logging

from src.app.core.warming_sql_queries import fetch_7d_transactions, fetch_lifetime_stats

BATCH_SIZE = 5000
WINDOW_7D = 604800
WINDOW_1H = 3600
WINDOW_24H = 86400


async def warm_up_redis(redis_client, db_connection):
    is_warmed = await redis_client.get("cache_status:warmed")
    if is_warmed:
        logging.info("Redis is already warmed up. Skipping...")
        return

    logging.info("Starting Redis Cache Warming...")
    pipe = redis_client.pipeline()
    commands_count = 0

    # =========================================================
    # ЭТАП 1: Загрузка глобальной статистики (Lifetime)
    # =========================================================
    rows_lifetime = await fetch_lifetime_stats(db_connection)

    for row in rows_lifetime:
        uid = row['uid1']
        key = f"user:{uid}:stats"

        pipe.hset(key, mapping={
            'last_tx_time': row['last_tx_time'],
            'tx_count': row['tx_count'],
            'sum_amt': row['sum_amt'],
            'sum_amt_sq': row['sum_amt_sq']
        })
        pipe.expire(key, WINDOW_24H * 180) # 180 days

        commands_count += 2
        if commands_count >= BATCH_SIZE:
            await pipe.execute()
            commands_count = 0

    if commands_count > 0:
        await pipe.execute()
        commands_count = 0

    # =========================================================
    # ЭТАП 2: Загрузка транзакций за 7 дней (Rolling Windows)
    # =========================================================
    rows_7d = await fetch_7d_transactions(db_connection)

    for row in rows_7d:
        uid = row['uid1']
        tx_time = row['TransactionDT']
        tx_id = row['TransactionID']
        tx_amt = row['TransactionAmt']
        device_sig = f"{row['DeviceType']}:{row['DeviceInfo']}"

        tx_key = f"user:{uid}:transactions"
        amt_key = f"user:{uid}:amounts_1h"
        dev_s_key = f"user:{uid}:devices_s"
        dev_z_key = f"user:{uid}:devices_z"

        # Восстанавливаем ZSET и SET
        pipe.zadd(tx_key, {str(tx_id): tx_time})
        pipe.zadd(amt_key, {f"{tx_id}:{tx_amt}": tx_time})
        pipe.sadd(dev_s_key, device_sig)
        pipe.zadd(dev_z_key, {device_sig: tx_time})

        # TTL
        pipe.expire(tx_key, WINDOW_7D)
        pipe.expire(amt_key, WINDOW_1H)  # amt_1h живет только час!
        pipe.expire(dev_s_key, WINDOW_24H * 180)
        pipe.expire(dev_z_key, WINDOW_24H * 180)

        commands_count += 8
        if commands_count >= BATCH_SIZE:
            await pipe.execute()
            commands_count = 0

    if commands_count > 0:
        await pipe.execute()

    await redis_client.set("cache_status:warmed", "1")
    logging.info("Cache Warming completed successfully!")
