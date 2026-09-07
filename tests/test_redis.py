import pytest
from fakeredis import FakeAsyncRedis

from src.app.routers import get_and_update_aggregates


@pytest.mark.asyncio
async def test_redis_aggregates():
    fake_redis = FakeAsyncRedis(decode_responses=True)

    res_1 = await get_and_update_aggregates(
        fake_redis,
        uid="123",
        transaction_id="12345",
        transaction_amt=100.0
    )

    # is_new_device_result, cnt_5m, cnt_1h, cnt_24h, cnt_7d, time_since_last_tx, avg_amt,
    # amt_vs_avg_ratio, std_amt, amt_1h, time_since_last_geo
    cnt_5m_first = res_1[1]
    assert cnt_5m_first == 1

    res_2 = await get_and_update_aggregates(
        fake_redis,
        uid="123",
        transaction_id="12346",
        transaction_amt=200.0
    )

    cnt_5m_second = res_2[1]
    assert cnt_5m_second == 2

    res_3 = await get_and_update_aggregates(
        fake_redis,
        uid="123",
        transaction_id="12347",
        transaction_amt=300.0
    )

    avg_amt = res_3[6]
    assert avg_amt == 150.0 # except last transaction. Also it's a feature leakage test

    amt_vs_avg_ratio = res_3[7]
    expected_ratio = 300.0 / (150.0 + 1)
    assert amt_vs_avg_ratio == pytest.approx(expected_ratio)
