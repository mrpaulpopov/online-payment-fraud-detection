import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()
API_KEY = os. environ.get("APP_API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key