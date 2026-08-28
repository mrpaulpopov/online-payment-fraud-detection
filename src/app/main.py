import json
import logging
import sys
from contextlib import asynccontextmanager
from sqlalchemy import text
import lightgbm as lgb
from fastapi import FastAPI

from src.app.cache_warmer import warm_up_redis
from src.app.core.data_loader_api import async_engine
from src.app.routers import router
from src.paths import LGBM_MODEL_PATH, INFERENCE_PATH

import redis.asyncio as redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading Redis")
    global redis_client
    redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
    await redis_client.ping()
    app.state.redis = redis_client
    logging.info("Successfully connected to Redis")

    logging.info("Starting up: Loading ML model...")
    try:
        # Critical health check
        if not LGBM_MODEL_PATH.exists():
            raise FileNotFoundError("LightGBM model is not found.")
        if not INFERENCE_PATH.exists():
            raise FileNotFoundError("Meta information is missing.")

        # Read JSON
        inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))

        model_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)

        app.state.model_lgbm = model_lgbm
        app.state.inference_meta = inference_meta

        logging.info("Models loaded successfully!")
    except Exception as e:
        logging.critical(f"Failed to load the model during startup: {e}")
        sys.exit(1)

    # -------------------------------
    # -------- Cache Warming --------
    # -------------------------------
    try:
        db_connection = await async_engine.connect()
        await warm_up_redis(redis_client, db_connection)
        await db_connection.close()
    except Exception as e:
        logging.error(f"Error during caching warm-up: {e}")

    yield
    logging.info("Shutting down: Flushing memory...")
    app.state.model_lgbm = None
    app.state.inference_meta = None
    await redis_client.close()


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
app.include_router(router)
