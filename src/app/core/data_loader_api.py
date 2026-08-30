# asynced version
import os
from sqlalchemy.ext.asyncio import create_async_engine
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DATABASE_URL = (
    f"postgresql+asyncpg://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}"
    f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'bankdb')}"
)

async_engine = create_async_engine(DATABASE_URL, echo=False)
