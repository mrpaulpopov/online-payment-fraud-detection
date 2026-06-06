from pathlib import Path

INFERENCE_PATH = Path("models/inference_meta.json")
INFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path("config.yaml")

LGBM_MODEL_PATH = Path("models/lgbm_model.txt")
NN_MODEL_PATH = Path("models/nn_model.pt")
IMPUTER_SCALER_PATH = Path("models/preprocessing.pkl")

PLOTS_DIR = Path("docs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)