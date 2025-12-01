from nodes.chat_node import ThinkpalmRAG, RetrieverNode
# Define pipeline state
from typing import TypedDict, List, Any, Dict
import os

from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
from datetime import datetime


# -----------------------
#  ChatState (BaseModel)
# -----------------------
class ChatState(BaseModel):
    # ---- Inputs ----
    question: str
    user_id: str
    vector_store_path: Optional[str] = None

    # ---- Retrieval ----
    retrieved_docs: List[Any] = Field(default_factory=list)
    filtered_docs: List[Any] = Field(default_factory=list)
    final_context: List[Any] = Field(default_factory=list)

    # ---- Initial Answer ----
    initial_answer: str = ""
    # rag_response: str = ""
    # answer: str = ""
    final_answer: str = ""

    # ---- Evaluation Fields ----
    graded_scores: List[Any] = Field(default_factory=list)
    # scores: List[float] = Field(default_factory=list)

    threshold_passed: bool = False
    
    eval_text: str = ""                          # <--- REQUIRED
    eval_score_faithfulness: float = 0.0         # <--- REQUIRED
    eval_score_relevance: float = 0.0            # <--- REQUIRED

    # ---- Regeneration ----
    regeneration_count: int = 0                  # <--- REQUIRED
    refined_question: Optional[str] = None       # <--- REQUIRED

    # ---- Memory ----
    chat_memory: Dict[str, List[tuple]] = Field(default_factory=dict)
    
    
# ------------- Add RAG node -------------
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")
CHAT_CONTAINER = os.getenv("CHAT_CONTAINER")



def rag_node(state: ChatState):
    """Single-node RAG pipeline"""
    user_id = state["user_id"]
    question = state["question"]
    answer = rag_instance.ask(user_id, question)
    state["rag_response"] = answer
    return state

rag_instance = ThinkpalmRAG()
chat_graph = StateGraph(ChatState)

# Instantiate node
shared_rag = rag_instance
# --- Define nodes ---
retriever_node = RetrieverNode(shared_rag)
chat_graph.add_node("retrieve", retriever_node.execute)

llm = shared_rag.llm
# chat_graph.add_node("evaluate", EvaluatorNode().execute)
# chat_graph.add_node("rerank", RerankNode().execute)
# chat_graph.add_node("regenerate", RegenerateNode(llm).execute)

# --- Define edges ---
chat_graph.add_edge(START, "retrieve")
chat_graph.add_edge("retrieve", END)
chat_graph = chat_graph.compile()



