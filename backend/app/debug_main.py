from fastapi import FastAPI
from app.config import checkpointer
from app.graph import rag_graph

api = FastAPI(title="Debug API")

@api.get("/health")
async def health_check():
    return {"status": "ok"}

@api.on_event("startup")
async def startup():
    print("DEBUG STARTUP - Initializing checkpointer")
    checkpointer.setup()
    print("DEBUG STARTUP SUCCESSFUL")
