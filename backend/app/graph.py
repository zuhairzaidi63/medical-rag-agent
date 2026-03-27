from langgraph.graph import StateGraph, END
from app.nodes import (
    AgentState, 
    medical_query_check, 
    query_optimization_node, 
    retrieval_node, 
    answer_generation_node
)
from app.config import checkpointer

def create_medical_rag_graph():
    workflow = StateGraph(AgentState)
        
    # Add the four nodes
    workflow.add_node("medical_query_check", medical_query_check)
    workflow.add_node("query_optimization", query_optimization_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("answer_generation", answer_generation_node)

    def route_query(state: AgentState) -> str:
        """Route based on whether the query is medical or not."""
        return "query_optimization" if state["is_medical_query"] else "end"

    # Start at the medical check
    workflow.set_entry_point("medical_query_check")
        
    # Conditional edge: medical queries proceed, others exit
    workflow.add_conditional_edges("medical_query_check", route_query, {
        "query_optimization": "query_optimization",
        "end": END
    })
        
    # The rest flow in sequence
    workflow.add_edge("query_optimization", "retrieval")
    workflow.add_edge("retrieval", "answer_generation")
    workflow.add_edge("answer_generation", END)

    # Compile with checkpointing enabled
    return workflow.compile(checkpointer=checkpointer)

# Create your graph
rag_graph = create_medical_rag_graph()
