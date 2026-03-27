import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from app.config import llm
from app.rag import hybrid_rag_query

class AgentState(TypedDict):
    """Shared state passed between every node in the graph."""
    messages: Annotated[List[BaseMessage], operator.add]      # Full conversation history 
    original_query: str        # Exactly what the user typed
    optimized_query: str       # Resolved, self-contained version for Qdrant
    retrieved_context: str     # Chunks from hybrid_rag_query()
    final_answer: str          # The response shown to the user
    is_medical_query: bool     # Did this pass the medical check?
    error_message: str         # Details if something went wrong
    session_id: str            # Links to PostgreSQL checkpoint

def medical_query_check(state: AgentState) -> AgentState:
    query = state["original_query"]
        
    prompt = """
    You are a query classifier for a Medical AI Assistant.
    Classify as MEDICAL if the query is about:
    - Medical conditions or diseases
    - Symptoms, diagnoses, or clinical findings
    - Treatments, medications, or procedures
    - Healthcare or public health topics
        
    Otherwise classify as NOT_MEDICAL.
    Respond with only: MEDICAL or NOT_MEDICAL
    """
        
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ])
        
    is_medical = (
        "MEDICAL" in response.content.upper() and
        "NOT_MEDICAL" not in response.content.upper()
    )
        
    return {
        "is_medical_query": is_medical,
        "error_message": (
            "This question doesn't appear to be medical. "
            "Please ask about a symptom, condition, or treatment."
            if not is_medical
            else state.get("error_message", "")
        )
    }

def query_optimization_node(state: AgentState) -> AgentState:
    query = state["original_query"]
    messages = state.get("messages", [])

    conversation_context = "\n".join([
        f"{msg.type}: {msg.content}"
        for msg in messages[-6:]
        if hasattr(msg, 'content')
    ])
        
    optimizer_prompt = f"""
    Rewrite the current query into a fully self-contained medical search string.
    Use the conversation history only if it adds essential context for the current question.
        
    Rules:
    - Return ONLY the rewritten query text.
    - Do NOT include any explanations, preambles, or "The rewritten query is...".
    - Do NOT mention the conversation history in your output.
    - If no rewriting is needed, return the original query exactly.
        
    Conversation history:
    {conversation_context}
        
    Current query: {query}
    Rewritten query:
    """
        
    response = llm.invoke([
        SystemMessage(content=optimizer_prompt),
        HumanMessage(content=query)
    ])
        
    optimized = response.content.strip()
        
    return {
        "optimized_query": optimized,
        "messages": [SystemMessage(content=f"Query optimized: {optimized}")]
    }

def retrieval_node(state: AgentState) -> AgentState:
    query = state.get("optimized_query", state.get("original_query", ""))
    context, ranked_results = hybrid_rag_query(
    query,
    top_k=16,
    rerank_top_k=5   # Reduced from 8 to 5 for better precision
)
        
    print(f"--- RETRIEVAL: Found context for '{query}' ({len(context)} chars) ---")
    return {
        "retrieved_context": context
    }

def answer_generation_node(state: AgentState) -> AgentState:
    query = state.get("optimized_query", state.get("original_query", ""))
    context = state.get("retrieved_context", "")
    messages = state.get("messages", [])

    conversation_history = "\n".join([
        f"{msg.type}: {msg.content}"
        for msg in messages[-10:]
        if hasattr(msg, 'content')
    ])
        
    system_prompt = """
    You are a Medical AI Assistant with access to a curated clinical knowledge base.
        
    RULES:
    1. Answer ONLY using information in the CONTEXT section below.
    2. Focus ONLY on the medical condition requested in the query. 
    3. If multiple conditions are in context, do NOT list symptoms for all of them unless explicitly asked.
    4. Never use external knowledge or make inferences beyond what context supports.
    5. If the context doesn't contain enough information to answer, say so explicitly.
        
    <CONVERSATION_HISTORY>
    {history}
    </CONVERSATION_HISTORY>
        
    <CONTEXT>
    {context}
    </CONTEXT>
    """
        
    response = llm.invoke([
        SystemMessage(content=system_prompt.format(
            history=conversation_history,
            context=context
        )),
        HumanMessage(content=query)
    ])
        
    return {
        "messages": [
            HumanMessage(content=query),
            SystemMessage(content=response.content)
        ],
        "final_answer": response.content
    }
