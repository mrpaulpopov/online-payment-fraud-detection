from fastapi import APIRouter, Depends, HTTPException, Response, status, Request
from sqlalchemy import text

from src.app.core.data_loader_api import async_engine
from src.app.schemas import Transaction
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
    try:
        model_pytorch = request.app.state.model_pytorch
    except Exception as e:
        model_pytorch = None

    if model_lgbm is None or model_pytorch is None:
        logging.error("ML models are not loaded into memory")
        health_status["models"] = "failed"
        is_healthy = False

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "error", "details": health_status}
    return {"status": "ok", "details": health_status}






@router.post("/predict")
async def predict_endpoint(data: Transaction, api_key: str = Depends(verify_api_key)):
    logging.info("Prediction request received")
    start = time.time()
    model_lgbm = request.app.state.model_lgbm
    model_pytorch = request.app.state.model_pytorch

    if model_lgbm is None or model_pytorch is None:
        return {
            "status": "error",
            "details": "ML models are not loaded into memory"
        }

    try:
        result = process_payment(data)
        logging.info(f"Prediction completed in {time.time() - start:.4f}s")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction": result}
