from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from nodes.chat_node import ThinkpalmRAG, RetrieverNode, GenerationNode, RerankNode, EvaluatorNode, RegenerateNode
from nodes.embedding_node import FAISSEmbeddingNode
from nodes.new_base_node import BaseNode
# Define pipeline state
from typing import TypedDict, List, Any, Dict
import os

class ChatState(TypedDict):
    question: str
    user_id: str
    vector_store_path: str
    retrieved_docs: List[Any]
    initial_answer: str
    
    graded_scores: List
    filtered_docs: List
    scores: List[float]
    # reranked_docs: List[Any]
    threshold_passed: bool
    rag_response: str
    
    final_context: List
    final_answer: str
    answer: str
    chat_memory: Dict[str, List[tuple]] 
    
# ------------- Add RAG node -------------
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")
CHAT_CONTAINER = os.getenv("CHAT_CONTAINER")

rag_instance = ThinkpalmRAG()

def rag_node(state: ChatState):
    """Single-node RAG pipeline"""
    user_id = state["user_id"]
    question = state["question"]
    answer = rag_instance.ask(user_id, question)
    state["rag_response"] = answer
    return state

chat_graph = StateGraph(ChatState)
'''
# --- Define nodes ---
chat_graph.add_node("retrieve", RetrieverNode().execute)
llm = RetrieverNode().rag_bot.llm
chat_graph.add_node("evaluate", EvaluatorNode().execute)
chat_graph.add_node("rerank", RerankNode().execute)
chat_graph.add_node("regenerate", RegenerateNode(llm).execute)

# --- Define edges ---
chat_graph.add_edge(START, "retrieve")
chat_graph.add_edge("retrieve", "evaluate")

# --- Conditional routing ---
def rerank_condition(state: dict):
    # If evaluation passes → skip rerank and regenerate → go to END
    return "rerank" if not state.get("threshold_passed") else "end"

chat_graph.add_conditional_edges(
    "evaluate",
    rerank_condition,
    {
        "rerank": "rerank",
        "end": END
    },
)

# After reranking → regenerate → end
chat_graph.add_edge("rerank", "regenerate")
chat_graph.add_edge("regenerate", END)

chat_graph = chat_graph.compile()

"""
# '''
# Instantiate node
chat_graph = StateGraph(ChatState)

# --- Define nodes ---
chat_graph.add_node("retrieve", RetrieverNode().execute)
llm = RetrieverNode().rag_bot.llm
# chat_graph.add_node("evaluate", EvaluatorNode().execute)
# chat_graph.add_node("rerank", RerankNode().execute)
# chat_graph.add_node("regenerate", RegenerateNode(llm).execute)

# --- Define edges ---
chat_graph.add_edge(START, "retrieve")
chat_graph.add_edge("retrieve", END)

# --- Conditiona

chat_graph = chat_graph.compile()


from IPython.display import Image, display
 
try:
    display(Image(chat_graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
 
# '''


