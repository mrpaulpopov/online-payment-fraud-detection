from pydantic import BaseModel

class PredictModel(BaseModel):
    user_str: str = "A3PHJ4NMHMBBUB"
    item_strs: list[str] = ["B000K8PH8C", "B001T6BK6M", "B007GFX0PY"]