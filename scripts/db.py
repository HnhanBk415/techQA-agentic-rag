import os
import psycopg
from dotenv import load_dotenv

def connect_db():
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("Thiếu DATABASE_URL trong .env")

    return psycopg.connect(database_url, connect_timeout=5)