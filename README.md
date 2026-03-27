# Medical RAG Agent 🏥🤖

A production-ready, clinical-grade AI assistant built with **LangGraph**, **FastAPI**, and **Streamlit**. This agent implements a hybrid RAG (Retrieval-Augmented Generation) pipeline with persistent memory and reranking for high-precision clinical answers.

## 🚀 Features
- **Hybrid Retrieval**: Combines semantic (Dense) and keyword-based (Sparse) search using Qdrant.
- **Persistent Conversation Memory**: Uses PostgreSQL and LangGraph checkpointers to track multi-turn clinical context across sessions.
- **Autonomous Reasoning**: Routes queries through specialized nodes for medical domain checking and query optimization.
- **Production Orchestration**: Fully containerized with Docker Compose for seamless deployment.
- **Modern Frontend**: Streamlit UI with real-time session tracking.

![Project Demo](demo.webp)

## 🛠️ Technical Stack
- **Graph Logic**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **Persistence**: [PostgreSQL](https://www.postgresql.org/)
- **Embeddings**: [TEI (Text Embeddings Inference)](https://github.com/huggingface/text-embeddings-inference) & Sparse Splade.
- **LLM**: Groq (Llama 3.3-70B)

## 📋 Prerequisites
- Docker & Docker Compose
- Groq API Key

## 🚦 Quick Start

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_actual_key_here
```

### 2. Launch the Stack
```bash
# Build and start services
docker compose up --build -d

# Verify services are healthy
curl http://localhost:8123/health
```

### 3. Usage
1. Open the UI: [http://localhost:8505](http://localhost:8505)
2. Use the **Admin Tools** sidebar to trigger the ingestion pipeline (populates the vector database).
3. Start a clinical conversation!

## 🔧 Project Structure
```text
medical-rag-agent/
├── backend/            # FastAPI & LangGraph logic
│   ├── app/
│   │   ├── nodes.py    # Agent reasoning nodes
│   │   ├── graph.py    # Workflow orchestration
│   │   └── rag.py      # Retrieval logic
├── frontend/           # Streamlit application
├── docker-compose.yaml # Service orchestration
└── requirements.txt    # Python dependencies
```

## 🔒 Security & Privacy
- This repository excludes sensitive medical data and environment variables via `.gitignore`.
- All model weights are cached locally in `./model_cache` to reduce external dependency dependencies.

## 📝 License
MIT License
