from fastapi import FastAPI
from src.app.routers import predict_endpoint
import logging
import lightgbm as lgb
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# Load model from file
fraud_model = lgb.Booster(model_file='models/lgbm_model.txt')
with open('models/inference_meta.json', 'r') as f:
    inference_meta = json.load(f)
    best_threshold = inference_meta['best_threshold']
logging.info("Model loaded.")

app = FastAPI()

app.include_router(predict_endpoint.router)

# uvicorn src.app.main:app --reload