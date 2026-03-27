import os
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import yaml

load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: {CONFIG_PATH} not found. Using hardcoded defaults.")
    config = {}

TEI_URL = config.get("services", {}).get("tei_url", "http://localhost:8080/embed")
QDRANT_URL = config.get("services", {}).get("qdrant_url", "http://localhost:6333")
DB_URI = config.get("services", {}).get("database_url", "postgresql://postgres:postgres@postgres:5432/checkpoints")
COLLECTION_NAME = config.get("collection", {}).get("name", "medical_rag_db")
DENSE_VECTOR_SIZE = config.get("collection", {}).get("dense_vector_size", 384)

from psycopg_pool import ConnectionPool

# Use a connection pool for a stable, long-running checkpointer
pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True})
checkpointer = PostgresSaver(pool)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
