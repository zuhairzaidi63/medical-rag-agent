import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
backend_endpoint = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide"
)
st.title("Medical RAG Chatbot")
st.markdown("Ask questions about medical conditions, symptoms, and treatments.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None   # None until the first query creates one
if "messages" not in st.session_state:
    st.session_state.messages = []       # Grows with every turn

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("Query Details"):
                st.json(message["metadata"])

if prompt := st.chat_input("Ask a medical question..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{backend_endpoint}/query",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id  # None on first turn
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Lock in the session ID from the first response
                    st.session_state.session_id = data["session_id"]
                    
                    if not data["is_medical"]:
                        st.warning("That doesn't appear to be a medical question.")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "That doesn't appear to be a medical question.",
                            "metadata": {
                                "Session ID": data["session_id"],
                                "Optimized Query": "",
                                "Domain Check": "Failed"
                            }
                        })
                    else:
                        st.markdown(data["answer"])
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "metadata": {
                                "Session ID": data["session_id"],
                                "Optimized Query": data["optimized_query"],
                                "Domain Check": "Passed" if data["is_medical"] else "Failed"
                            }
                        })
                else:
                    st.error(f"Backend returned {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

with st.sidebar:
    st.header("Admin Tools")
    
    if st.button("Run Ingestion Pipeline"):
        with st.spinner("Ingesting documents... this may take a few minutes."):
            try:
                response = requests.post(f"{backend_endpoint}/ingest", timeout=300)
                if response.status_code == 200:
                    st.success("Knowledge base updated successfully!")
                else:
                    st.error(f"Ingestion failed: {response.text}")
            except Exception as e:
                st.error(f"Could not reach backend: {e}")
    
    st.markdown("---")
    st.header("How It Works")
    st.markdown("""
    - **LangGraph** routes and manages each turn
    - **Hybrid RAG** (dense + sparse) retrieves from Qdrant
    - **Cross-encoder reranking** improves precision
    - **PostgreSQL** persists every session
    """)
    
    st.markdown("---")
    st.header("Try This Conversation")
    st.markdown("""
    1. *"What are the symptoms of pneumonia in elderly patients?"*
    2. *"What causes it?"*
    3. *"What's the first-line treatment?"*
    4. *"What if they're penicillin-allergic?"*
    """)
