from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
import tiktoken
import os

db_name = os.getenv("EMBEDDING_DB_NAME", "openai_embeddings/thinkpalm")
model_name = os.getenv("MODEL_NAME", "nomic-embed-text")
embeddings = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")

class FAISSEmbeddingHandler:
    def __init__(self, model_name=model_name, base_url="http://localhost:11434", db_path=db_name):
        self.model_name = model_name
        self.base_url = base_url
        self.db_path = db_path
        self.embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def load_markdown(self, file_path: str) -> str:
        """Read a markdown file and return its content as string."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def create_chunks(self, text: str, chunk_size=1000, chunk_overlap=100):
        """Split text into manageable chunks."""
        lc_docs = [LCDocument(text)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(lc_docs)

    def create_faiss_index(self, file_path: str):
        """Create FAISS vector store from a markdown file."""
        print(f"📄 Loading markdown: {file_path}")
        content = self.load_markdown(file_path)
        chunks = self.create_chunks(content)
        print(f"🧩 Created {len(chunks)} chunks")

        print(f"⚙️ Generating embeddings with model: {self.model_name}")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        print(f"💾 Saving FAISS index to {self.db_path}")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        vector_store.save_local(self.db_path)
        print("✅ Vector store saved successfully!")

    def load_faiss_index(self):
        """Load an existing FAISS vector store."""
        print(f"📂 Loading FAISS index from {self.db_path}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"No FAISS index found at {self.db_path}")

        new_vector_store = FAISS.load_local(
            self.db_path, 
            embeddings=self.embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS index loaded successfully!")
        return new_vector_store


if __name__ == "__main__":
    handler = FAISSEmbeddingHandler()
    
    # Step 1: Create the FAISS index from a markdown file
    handler.create_faiss_index("chains/embedding_input.md")

    # Step 2: Load the FAISS index for later use
    vector_store = handler.load_faiss_index()
    print(vector_store)
