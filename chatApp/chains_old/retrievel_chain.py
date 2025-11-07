import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()


class ThinkpalmRAG:
    def __init__(self):
        # ========== CONFIG ==========
        self.cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("COSMOS_KEY")
        self.db_name = os.getenv("COSMOS_DATABASE")
        self.container_name = os.getenv("COSMOS_CONTAINER")
        self.history_container_name = os.getenv("CHAT_CONTAINER")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "text-embedding-3-small")

        self.top_k = 5
        self.memory_store = {}

        # ========== CLIENTS ==========
        self.client = CosmosClient(url=self.cosmos_endpoint, credential=self.cosmos_key)
        self.db = self._ensure_database(self.db_name)
        self.container = self._ensure_container(self.container_name)
        self.history_container = self._ensure_container(self.history_container_name, partition_key="user_id")

        # ========== EMBEDDINGS + LLM ==========
        self.embeddings = OpenAIEmbeddings(model=self.model_name)
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=self.OPENAI_API_KEY,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        # ========== RETRIEVER & RAG CHAIN ==========
        self.rag_chain = self._build_rag_chain()

    # ------------------------------------------------------------
    # COSMOS HELPERS
    # ------------------------------------------------------------
    def _ensure_database(self, db_name):
        try:
            return self.client.create_database_if_not_exists(id=db_name)
        except Exception as e:
            print(f"Error ensuring database: {e}")
            raise

    def _ensure_container(self, container_name, partition_key="id"):
        try:
            return self.db.create_container_if_not_exists(
                id=container_name, partition_key=PartitionKey(path=f"/{partition_key}")
            )
        except Exception as e:
            print(f"Error ensuring container '{container_name}': {e}")
            raise

    # ------------------------------------------------------------
    # VECTOR SEARCH ON COSMOS
    # ------------------------------------------------------------
    def _embed_query(self, query: str):
        """Get OpenAI embeddings for the query."""
        return self.embeddings.embed_query(query)

    def search_cosmos_documents(self, query: str):
        """Perform vector search using Cosmos SQL API's VectorDistance function."""
        query_emb = self._embed_query(query)
        emb_json = json.dumps(query_emb)

        sql_query = f"""
        SELECT TOP {self.top_k}
            c.id, c.text, c.metadata,
            VectorDistance(c.vector_embedding, {emb_json}) AS score
        FROM c
        ORDER BY VectorDistance(c.vector_embedding, {emb_json})
        """

        try:
            results = list(self.container.query_items(query=sql_query, enable_cross_partition_query=True))
        except Exception as e:
            print(f"❌ Error querying Cosmos DB: {e}")
            results = []

        # Convert to LangChain-like docs
        docs = [
            {
                "id": d.get("id"),
                "page_content": d.get("text", ""),
                "metadata": d.get("metadata", {}),
                "score": float(d.get("score", 0)),
            }
            for d in results
        ]
        print(f"✅ Retrieved {len(docs)} relevant docs from Cosmos DB.")
        return docs

    # ------------------------------------------------------------
    # PROMPT + RAG CHAIN
    # ------------------------------------------------------------
    @staticmethod
    def format_docs(docs):
        """Combine retrieved docs into one context block."""
        return "\n\n".join([doc["page_content"] for doc in docs])

    def _build_prompt_template(self):
        """Your structured business prompt template."""
        return ChatPromptTemplate.from_template("""
You are a policy assistant for MOL Chemical Tankers. 
Answer strictly based on the provided context.

Question: {question}
Context: {context}
Answer:
""")

    
    def _build_rag_chain(self):
        """Create the retrieval + generation chain."""
        prompt = self._build_prompt_template()

        # Wrap the Cosmos search in a RunnableLambda so it can be piped
        retrieve_runnable = RunnableLambda(self.search_cosmos_documents)

        # Combine into a proper RAG chain
        return (
            {
                "context": retrieve_runnable | RunnableLambda(self.format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------
    # ASK (MAIN ENTRYPOINT)
    # ------------------------------------------------------------
    def ask(self, question: str):
        """Run full RAG flow: retrieve from Cosmos + generate answer."""
        print(f"💬 Asking: {question}")
        response = self.rag_chain.invoke(question)
        docs = self.search_cosmos_documents(question)
        return response, docs
