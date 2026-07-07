# asynced version
import os
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = (
    f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

async_engine = create_async_engine(DATABASE_URL, echo=False)
