import os
import json
from pathlib import Path
from typing import Dict, Any
from embeddings import FAISSEmbeddingHandler, EMBEDDING_BACKEND, DEFAULT_DB_SUBPATH
from langchain.schema import HumanMessage, AIMessage

class UserManager:
    def __init__(self, data_root: Path, embedding_backend: str = "ollama"):
        self.root = Path(data_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = embedding_backend
        # keep in-memory handlers per user
        self._handlers = {}
        # simple in-memory user metadata cache
        self._users_meta = {}

    def ensure_user(self, email: str):
        user_folder = self.get_user_folder(email)
        user_folder.mkdir(parents=True, exist_ok=True)
        (user_folder / "uploads").mkdir(exist_ok=True)
        (user_folder / "docs_markdown").mkdir(exist_ok=True)
        (user_folder / "vectorstores").mkdir(exist_ok=True)
        meta = user_folder / "meta.json"
        if not meta.exists():
            meta.write_text(json.dumps({"files": [], "history": []}, indent=2))
        return user_folder

    def get_user_folder(self, email: str) -> Path:
        safe = email.replace("@", "_at_").replace(".", "_")
        return self.root / safe

    def get_embedding_handler(self, email: str) -> FAISSEmbeddingHandler:
        user_folder = self.get_user_folder(email)
        db_root = user_folder / DEFAULT_DB_SUBPATH
        if email in self._handlers:
            return self._handlers[email]
        handler = FAISSEmbeddingHandler(db_root, backend=self.backend)
        self._handlers[email] = handler
        return handler

    def add_uploaded_file(self, email: str, raw_path: str, md_path: str):
        meta = self._read_meta(email)
        meta["files"].append({"raw": raw_path, "md": md_path})
        self._write_meta(email, meta)

    def list_user_files(self, email: str):
        meta = self._read_meta(email)
        return meta.get("files", [])


    def append_history(self, email: str, record: Dict[str, Any]):
        """Append a chat turn (question-answer) to the user's meta.json."""
        meta = self._read_meta(email)
        history = meta.setdefault("chat_history", [])

        # Convert to role-based message pairs
        if "question" in record:
            history.append({"role": "human", "content": str(record["question"])})
        if "answer" in record:
            history.append({"role": "ai", "content": str(record["answer"])})

        # Save
        self._write_meta(email, meta)


    def get_history(self, email: str):
        """Load last N messages as HumanMessage/AIMessage objects."""
        meta = self._read_meta(email)
        history = meta.get("chat_history", [])

        messages = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
        return messages

    def _read_meta(self, email: str):
        path = self.get_user_folder(email) / "meta.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_meta(self, email: str, meta: dict):
        import json
        from langchain.schema import HumanMessage, AIMessage

        meta_path = self.get_user_folder(email) / "meta.json"

        def safe_serialize(obj):
            """Recursively make all objects JSON-serializable."""
            if isinstance(obj, (HumanMessage, AIMessage)):
                return {"role": obj.type, "content": obj.content}
            elif isinstance(obj, list):
                return [safe_serialize(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: safe_serialize(v) for k, v in obj.items()}
            else:
                # Try to convert other objects safely
                try:
                    json.dumps(obj)
                    return obj
                except TypeError:
                    return str(obj)

        # Apply to full metadata (not just chat_history)
        meta_serialized = safe_serialize(meta)

        meta_path.write_text(json.dumps(meta_serialized, indent=2), encoding="utf-8")

    # lightweight RAG builder per request (wrap your existing ThinkpalmRAG)
    def get_rag_for_user(self, email: str, handler: FAISSEmbeddingHandler, vectorstore, chat_history = None):
        # Build a quick RAG wrapper that reuses your existing ThinkpalmRAG class structure
        from chains.retrievel_chain import ThinkpalmRAG  # your existing chain adapted earlier
        rag = ThinkpalmRAG(embeddings=handler.embeddings, vector_store=vectorstore, chat_history=chat_history)
        return rag

from fastapi import Header, HTTPException

def get_user_from_header(x_user_email: str = Header(None)) -> str:
    """
    Extracts the user's email from the request header.
    Expected header: X-User-Email: user@example.com
    """
    if not x_user_email:
        raise HTTPException(status_code=400, detail="Missing X-User-Email header")
    return x_user_email