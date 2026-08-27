import time

WINDOW_7D = 604800
WINDOW_24H = 86400
WINDOW_1H = 3600
WINDOW_5M = 300


async def get_and_update_aggregates(redis_client,
                                    uid: str,
                                    transaction_id: str,
                                    transaction_amt: str,
                                    deviceinfo: str,
                                    devicetype: str) -> tuple[
    int, int, int, int, int, float, float, float, float, float, float]:
    now = time.time()
    key = f"user:{uid}"

    pipe = redis_client.pipeline()

    # =====================================
    # -------- TIME SINCE LAST TX ---------
    # =====================================
    hash_uid_key = f'{key}:stats'
    pipe.hget(hash_uid_key, 'last_tx_time')  # [0]
    pipe.hset(hash_uid_key, 'last_tx_time', now)  # [1]

    # =====================================
    # -------HASH LIFETIME AGGREGATES -----
    # =====================================
    pipe.hincrby(hash_uid_key, 'tx_count', 1)  # [2]
    pipe.hincrbyfloat(hash_uid_key, 'sum_amt', float(transaction_amt))  # [3]
    pipe.hincrbyfloat(hash_uid_key, 'sum_amt_sq', float(transaction_amt) ** 2)  # [4]

    # =====================================
    # -------- IS_NEW_DEVICE_UID1 ---------
    # =====================================
    devices_s_key = f'{key}:devices_s'
    devicetype_signature = f'{devicetype}:{deviceinfo}'
    pipe.sismember(devices_s_key, devicetype_signature)  # [5]
    pipe.sadd(devices_s_key, devicetype_signature)  # [6]

    # =====================================
    # ----- TIME SINCE LAST GEO CHANGE ----
    # =====================================
    devices_z_key = f'{key}:devices_z'  # Z key
    pipe.zadd(devices_z_key, {f'{devicetype_signature}': now})  # [7]
    pipe.zrange(devices_z_key, 0, 1, desc=True, withscores=True)  # [8]

    # =====================================
    # ---------- ROLLING WINDOWS ----------
    # =====================================
    tx_key = f'{key}:transactions'
    pipe.zremrangebyscore(tx_key, 0, now - WINDOW_7D)  # [9]
    pipe.zadd(tx_key, {transaction_id: now})  # TX-TIME # [10]
    pipe.zcount(tx_key, now - WINDOW_5M, now)  # [11]
    pipe.zcount(tx_key, now - WINDOW_1H, now)  # [12]
    pipe.zcount(tx_key, now - WINDOW_24H, now)  # [13]
    pipe.zcount(tx_key, now - WINDOW_7D, now)  # [14]

    # =====================================
    # -------------- AMT_1H ---------------
    # =====================================
    amt_key = f'{key}:amounts_1h'
    pipe.zadd(amt_key, {f'{transaction_id}:{transaction_amt}': now})  # [15]
    pipe.zremrangebyscore(amt_key, 0, now - WINDOW_1H)  # [16]
    pipe.zrange(amt_key, now - WINDOW_1H, now, byscore=True)  # [17]

    # =====================================
    # --------------- TTL -----------------
    # =====================================
    pipe.expire(tx_key, WINDOW_7D)
    pipe.expire(amt_key, WINDOW_1H)
    pipe.expire(devices_s_key, WINDOW_7D)
    pipe.expire(devices_z_key, WINDOW_7D)
    pipe.expire(hash_uid_key, WINDOW_7D)

    # =====================================
    # ------------- RESULTS ---------------
    # =====================================
    results = await pipe.execute()

    # time_since_last_tx
    if results[0] is not None:
        old_tx_time = float(results[0])
        time_since_last_tx = now - old_tx_time
    else:
        time_since_last_tx = 0

    # avg_amt, std_amt
    tx_count = int(results[2])
    sum_amt = float(results[3])
    sum_amt_sq = float(results[4])

    # (1 PRECEDING)
    old_tx_count = tx_count - 1
    old_sum_amt = sum_amt - float(transaction_amt)
    old_sum_amt_sq = sum_amt_sq - (float(transaction_amt) ** 2)

    if old_tx_count > 0:
        avg_amt = old_sum_amt / old_tx_count
        variance = (old_sum_amt_sq / old_tx_count) - (avg_amt ** 2)  # Var = (Sum(x^2) / N) - Mean^2
        std_amt = variance ** 0.5 if variance > 0 else 0.0
    else:
        avg_amt = 0.0
        std_amt = 0.0

    # amount/average ratio
    amt_avg_ratio = float(transaction_amt) / avg_amt if avg_amt else 0

    is_new_device_result = 1 - results[5]

    # time_since_last_geo_change
    raw_time_from_last_geo = results[8]
    if len(raw_time_from_last_geo) > 1:
        _, last_other_geo_time = raw_time_from_last_geo[1]
        time_since_last_geo = now - last_other_geo_time
    else:
        time_since_last_geo = 0.0

    cnt_5m = results[11]
    cnt_1h = results[12]
    cnt_24h = results[13]
    cnt_7d = results[14]

    # amt_1h. O(n), but there are very few items.
    amt_1h = 0.0
    raw_amt_1h = results[17]
    for item in raw_amt_1h:
        _, amt = item.split(':')
        amt_1h += float(amt)

    # DEBUG
    # print('-------- NEW TX --------')
    # print('Is new device:')
    # print(is_new_device_result)
    # print('cnt_5m, cnt_1h, cnt_24h, cnt_7d:')
    # print(cnt_5m, cnt_1h, cnt_24h, cnt_7d)
    # print('time_since_last_tx:')
    # print(time_since_last_tx)
    # print('avg_amt:')
    # print(avg_amt)
    # print('amt_avg_ratio:')
    # print(amt_avg_ratio)
    # print('std_amt')
    # print(std_amt)
    # print('amt_1h:')
    # print(amt_1h)
    # print('time_since_last_geo')
    # print(time_since_last_geo)
    # print()

    return (is_new_device_result, cnt_5m, cnt_1h, cnt_24h, cnt_7d,
            time_since_last_tx, avg_amt, amt_avg_ratio, std_amt, amt_1h, time_since_last_geo)
