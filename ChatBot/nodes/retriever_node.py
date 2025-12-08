# nodes/retriever_node.py
from nodes.base_node import BaseNode
from collections import deque
from datetime import datetime

class RetrieverNode(BaseNode):
    def __init__(self, rag):
        super().__init__("RetrieverNode")
        self.rag = rag

    def execute(self, state: dict) -> dict:
        """
        Read: refined_question
        Writes: retrieved_docs (list of Document), retrieval_trace (optional)
        Does NOT call rag.ask() or cause LLM generation.
        """
        user_id = state.get("user_id")
        q = state.get("refined_question", "").strip()

        # Ensure user history slot exists
        if user_id not in self.rag.chat_memory:
            self.rag.chat_memory[user_id] = deque(maxlen=self.rag.history_limit)

        # Enrich query with hints to improve retrieval accuracy
        hinted = self.rag._add_retrieval_hints(q)

        docs = self.rag.search_cosmos_documents(hinted)
        
        for i, d in enumerate(docs):
                preview = d.page_content.replace("\n", " ").strip()[:180]
                score = d.metadata.get("score", 0)
                print(f"[DEBUG] Doc {i+1} | Score: {score:.3f} | Snippet: {preview}...")

        # Save full retrieval trace for external inspection
        with open("Verification_retrieved_docs.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n==============================\n")
            f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
            f.write(f"Question: {q}\n")
            for i, d in enumerate(docs):
                text = d.page_content
                score = d.metadata.get("score", 0)
                f.write(f"Doc {i+1} (Score {score:.3f}):\n{text}\n\n")
            f.write(f"Hinted query: {hinted}\n")

        # Optional: store a small trace for debugging
        trace = {
            "query": q,
            "hinted_query": hinted,
            "docs_count": len(docs)
        }

        return {
            "retrieved_docs": docs,
            "retrieval_trace": trace
        }
