from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.rag import ingestion_pipeline
from pydantic import BaseModel
from app.config import checkpointer
from app.graph import rag_graph
from typing import Optional
import uuid

api = FastAPI(title="Medical RAG Agent API")

@api.on_event("startup")
async def startup_event():
    print("--- STARTUP: Initializing services ---")
    try:
        # Ensure PostgreSQL tables are created
        checkpointer.setup()
        print("--- STARTUP: Database setup complete ---")
    except Exception as e:
        print(f"--- STARTUP ERROR: {e} ---")
        raise e

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # If None, generate a fresh one

class QueryResponse(BaseModel):
    answer: str
    optimized_query: str
    is_medical: bool
    error_message: str = ""
    session_id: str          # Always return this; the client needs it for continuity

@api.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # If the client doesn't have a session yet, start one
        session_id = request.session_id or str(uuid.uuid4())
        
        # This is how LangGraph knows which PostgreSQL checkpoint to load
        config = {"configurable": {"thread_id": session_id}}
        
        initial_state = {
            "messages": [],
            "original_query": request.query,
            "optimized_query": "",
            "retrieved_context": "",
            "final_answer": "",
            "is_medical_query": True,
            "error_message": "",
            "session_id": session_id
        }
        
        # Run the full four-node pipeline from Part 2
        result = rag_graph.invoke(initial_state, config)
        
        # Handle off-topic queries gracefully
        if not result["is_medical_query"]:
            return QueryResponse(
                answer=result.get("error_message", "Query rejected"),
                optimized_query="",
                is_medical=False,
                error_message=result.get("error_message", ""),
                session_id=session_id
            )
        
        return QueryResponse(
            answer=result["final_answer"],
            optimized_query=result["optimized_query"],
            is_medical=result["is_medical_query"],
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/health")
async def health_check():
    """Confirm the service is up and the checkpointer is connected."""
    return {"status": "healthy", "checkpointer": "connected"}

@api.post("/ingest")
async def run_ingestion():
    """Re-trigger the Part 1 ingestion pipeline to refresh the knowledge base."""
    try:
        ingestion_pipeline()
        return {"status": "success", "message": "Ingestion completed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
