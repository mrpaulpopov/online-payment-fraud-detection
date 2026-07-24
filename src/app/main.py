import json
import logging
import sys
from contextlib import asynccontextmanager

import lightgbm as lgb
from fastapi import FastAPI

from src.app.routers import router
from src.paths import IMPUTER_SCALER_PATH, LGBM_MODEL_PATH, INFERENCE_PATH, NN_MODEL_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up: Loading ML model...")
    try:
        # Critical health check
        if not LGBM_MODEL_PATH.exists():
            raise FileNotFoundError("LightGBM model is not found.")
        if not INFERENCE_PATH.exists():
            raise FileNotFoundError("Meta information is missing.")

        # Read JSON
        inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))

        # ----------- LGBM Side ----------
        model_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)


        app.state.model_lgbm = model_lgbm
        app.state.inference_meta = inference_meta

        logging.info("Models loaded successfully!")
    except Exception as e:
        logging.critical(f"Failed to load the model during startup: {e}")
        sys.exit(1)
    yield
    logging.info("Shutting down: Flushing memory...")
    app.state.model_lgbm = None
    app.state.inference_meta = None

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
app.include_router(router)
