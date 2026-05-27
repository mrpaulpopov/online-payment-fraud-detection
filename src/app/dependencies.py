from fastapi import HTTPException, Header


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "123":
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key