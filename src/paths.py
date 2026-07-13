from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

INFERENCE_PATH = PROJECT_ROOT / "models/inference_meta.json"
INFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)

LGBM_MODEL_PATH = PROJECT_ROOT/ "models/lgbm_model.txt"
NN_MODEL_PATH = PROJECT_ROOT / "models/nn_model.pt"
IMPUTER_SCALER_PATH = PROJECT_ROOT / "models/imputerscaler.pkl"

CACHE_DIR = PROJECT_ROOT / "models/datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = PROJECT_ROOT / "docs/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(PROJECT_ROOT) # debug