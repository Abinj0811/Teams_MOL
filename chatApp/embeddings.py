import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from langchain_community.vectorstores import FAISS
import os
import shutil

# We support two backends: Ollama (local) and OpenAI (cloud)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "ollama").lower()
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
DEFAULT_DB_SUBPATH = "vectorstores/faiss_index/thinkpalm"  # used as folder name template

# lazy imports
def _ollama_embeddings(model_name="nomic-embed-text"):
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model_name, base_url=OLLAMA_BASE)

def _openai_embeddings(model_name="text-embedding-3-small"):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name)

class FAISSEmbeddingHandler:
    """
    Create / load per-user FAISS index from markdown files.
    Stores vectorstore in the provided db_path (folder).
    """
    def __init__(self, db_root: Path, backend: str = EMBEDDING_BACKEND, model_name: str = None):
        self.db_root = Path(db_root)
        self.backend = backend
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = "nomic-embed-text" if backend == "ollama" else "text-embedding-3-small"

        if backend == "ollama":
            self.embeddings = _ollama_embeddings(self.model_name)
        else:
            self.embeddings = _openai_embeddings(self.model_name)

        # ensure db root exists
        self.db_root.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.db_root)

    def create_faiss_index(self, file_path: str):
        # file_path should be path to markdown text file
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document as LCDocument

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lc_docs = [LCDocument(content)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(lc_docs)
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        # save local
        vector_store.save_local(self.db_path)

    def load_faiss_index(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"No FAISS index at {self.db_path}")
        new_vector_store = FAISS.load_local(self.db_path, embeddings=self.embeddings, allow_dangerous_deserialization=True)
        return new_vector_store
