import json

from fastapi import APIRouter, Depends, HTTPException, Response, status, Request
from sqlalchemy import text

from src.app.core.data_loader_api import async_engine
from src.redis.redis_utils import get_and_update_aggregates
from src.app.schemas import Transaction, PredictionResponse
from src.app.dependencies import verify_api_key
from src.app.services import process_payment
import logging
import time

router = APIRouter()


@router.get("/healthcheck")
async def healthcheck_endpoint(response: Response, request: Request):
    health_status = {
        "api": "ok",
        "database": "ok",
        "models": "ok"
    }
    is_healthy = True

    try:
        async with async_engine.connect() as connection:
            await connection.execute(text(f"SELECT 1;"))
    except Exception as e:
        logging.error(f"Database healthcheck failed: {e}")
        health_status["database"] = "failed"
        is_healthy = False

    try:
        model_lgbm = request.app.state.model_lgbm
    except Exception as e:
        model_lgbm = None

    if model_lgbm is None:
        logging.error("ML model are not loaded into memory")
        health_status["models"] = "failed"
        is_healthy = False

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "error", "details": health_status}
    return {"status": "ok", "details": health_status}


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(data: Transaction, request: Request,
                           api_key: str = Depends(verify_api_key)):
    logging.info("Prediction request received")
    start = time.time()
    model_lgbm = request.app.state.model_lgbm
    inference_meta = request.app.state.inference_meta
    redis_client = request.app.state.redis

    if model_lgbm is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded into memory")

    if inference_meta is None:
        raise HTTPException(status_code=503, detail="Insufficient metadata for an inference")

    try:
        transaction_dict = data.model_dump()  # convert Pydantic to DataFrame

        # ==== REDIS: CREATING AGGREGATES =====
        (is_new_device_result, cnt_5m, cnt_1h, cnt_24h,
         cnt_7d, time_since_last_tx, avg_amt, amt_vs_avg_ratio,
         std_amt, amt_1h,
         time_since_last_geo) = await get_and_update_aggregates(redis_client, uid=transaction_dict["card1"],
                                                                transaction_id=transaction_dict["TransactionID"],
                                                                transaction_amt=transaction_dict["TransactionAmt"],
                                                                deviceinfo=transaction_dict["DeviceInfo"],
                                                                devicetype=transaction_dict["DeviceType"]
                                                                )
        transaction_dict["is_new_device_uid1"] = is_new_device_result
        transaction_dict["cnt_5m"] = cnt_5m
        transaction_dict["cnt_1h"] = cnt_1h
        transaction_dict["cnt_24h"] = cnt_24h
        transaction_dict["cnt_7d"] = cnt_7d
        transaction_dict["time_since_last_tx"] = time_since_last_tx
        transaction_dict["avg_amt_per_uid1"] = avg_amt
        transaction_dict["amt_vs_avg_ratio"] = amt_vs_avg_ratio
        transaction_dict["std_amt_per_uid1"] = std_amt
        transaction_dict["amt_1h"] = amt_1h
        transaction_dict["time_since_last_geo_change"] = time_since_last_geo
        # =======================================

        # ======== SEND TO A SERVICE ============
        transaction_id, is_fraud, fraud_probability, reason = process_payment(transaction_dict, inference_meta, model_lgbm)

        # ======= REDIS: SAVE TO SQL ============
        await redis_client.rpush("manual_tx_queue", json.dumps(transaction_dict))

        latency = round(float((time.time() - start)) * 1000, 2)
        logging.info(f"Prediction completed in {latency:.8f}s")
    except HTTPException as e:
        raise e
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=str(e)) # TODO
    return {"transaction_id": int(transaction_id),
            "is_fraud": bool(is_fraud),
            "fraud_probability": float(fraud_probability) if fraud_probability is not None else None,
            "action": "BLOCK" if is_fraud else "APPROVE",
            "reason": str(reason),  # explicitly convert types for all of 4 outputs
            "latency_ms": float(latency)
            }
