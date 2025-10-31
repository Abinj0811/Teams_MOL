from langgraph.graph import StateGraph, START, END
from nodes.new_doc_loader import DocumentLoaderNode
from nodes.new_doc_processor_node import DocumentProcessorNode
from nodes.embedding_node import FAISSEmbeddingNode
from typing import TypedDict
# Define state schema as a list of tuples: (key, type)


class state_schema(TypedDict):
    document_paths: list
    saved_documents: str
    processed_results: list
    

# Instantiate node
doc_loader = DocumentLoaderNode()  # uses default ./saved_docs
doc_processor = DocumentProcessorNode()
embedder = FAISSEmbeddingNode()
# Create graph
# ❌ Change: Pass the dictionary directly as the schema
graph = StateGraph(state_schema)

# Add node
graph.add_node("load_documents", doc_loader.execute)
graph.add_node("process_documents", doc_processor.execute)
graph.add_node("embedding", embedder.execute)
# Connect START → node → END
graph.add_edge(START, "load_documents")
graph.add_edge("load_documents", "process_documents")
graph.add_edge("process_documents", "embedding")
graph.add_edge("embedding", END)

# Compile graph
"""
graph.add_edge(START, "embedding")
graph.add_edge("embedding", END)
"""
graph = graph.compile()