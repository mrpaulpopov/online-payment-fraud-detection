import json
import logging
import pickle
from contextlib import asynccontextmanager

import lightgbm as lgb
import torch
from fastapi import FastAPI

from src.app.routers import router
from src.models.autoencoder import autoencoder_nn
from src.paths import IMPUTER_SCALER_PATH, LGBM_MODEL_PATH, INFERENCE_PATH, NN_MODEL_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up: Loading ML models...")
    try:
        # Read JSON
        inference_meta = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
        input_dim = inference_meta["input_dim"]
        latent_dim = inference_meta["latent_dim"]

        # Read Imputer, Scaler
        with IMPUTER_SCALER_PATH.open('rb') as f:
            imp_object = pickle.load(f)
        num_imputer = imp_object['imputer']
        scaler = imp_object['scaler']

        # --------- PyTorch Side ---------
        model_pytorch = autoencoder_nn(input_dim, latent_dim)
        model_pytorch.load_state_dict(torch.load(NN_MODEL_PATH, weights_only=True))
        model_pytorch.eval()

        # ----------- LGBM Side ----------
        model_lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)


        app.state.model_lgbm = model_lgbm
        app.state.num_imputer = num_imputer
        app.state.scaler = scaler
        app.state.model_pytorch = model_pytorch
        app.state.inference_meta = inference_meta

        logging.info("Models loaded successfully!")
    except Exception as e:
        logging.error(f"Failed to load models during startup: {e}")
        raise e
    yield
    logging.info("Shutting down: Flushing memory...")
    app.state.model_lgbm = None
    app.state.num_imputer = None
    app.state.scaler = None
    app.state.model_pytorch = None
    app.state.inference_meta = None

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
app.include_router(router)
