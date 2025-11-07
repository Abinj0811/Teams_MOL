# from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# from langchain_community import hub #TO PULL RAG PROMPTS
from langchain_core.output_parsers import StrOutputParser # GETTING FINAL OUT AS STRING
from langchain_core.runnables import RunnablePassthrough #parse question and context directly to LLM
from langchain_core.prompts import ChatPromptTemplate #to pass prompt with context (chunk of data)

# from langchain_ollama import ChatOllama
from langchain_core.callbacks import StreamingStdOutCallbackHandler

class ThinkpalmRAG:
    def __init__(self):
        # Load environment variables
        # self.db_name = os.getenv("EMBEDDING_DB_NAME")
        # self.model_name = os.getenv("MODEL_NAME")
        self.db_name = os.getenv("EMBEDDING_DB_NAME", "./chatApp/openai_embeddings/MOL_openai_H")
        self.model_name = os.getenv("MODEL_NAME", "nomic-embed-text")

        self.base_url = "http://localhost:11434"
        print("Using model:", self.model_name, "with DB:", self.db_name, "at", self.base_url)
        # self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.no_of_docs = 4
        # self.embeddingmodel_name = "nomic-embed-text"  # or "gpt-3.5-turbo", etc.
        # Initialize embeddings and vectorstore
        # self.embeddings = OllamaEmbeddings(model=self.embeddingmodel_name, base_url=self.base_url)
        self.embeddings = OpenAIEmbeddings(model=self.model_name, api_key=self.api_key)
        
        # self.embeddings = OpenAIEmbeddings(model=self.model_name)

        self.vector_store = self._load_vector_store()

        # Initialize chat model
        # self.model = ChatGroq(
        #     model="llama-3.1-8b-instant",
        #     api_key=self.groq_api_key,
        #     streaming=True,
        #     callbacks=[StreamingStdOutCallbackHandler()],
        # )

        self.model = ChatOpenAI(
            model="gpt-4.1",  # or "gpt-4o", "gpt-3.5-turbo", etc.
            api_key=self.api_key,  # rename or reuse your existing key variable
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        
        # Initialize retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': self.no_of_docs, 'fetch_k': 50, 'lambda_mult': 1}
        )
        
        # Create the RAG chain
        self.rag_chain = self._build_rag_chain()

    def _load_vector_store(self):
        """Load FAISS vector store."""
        if not os.path.exists(self.db_name):
            raise FileNotFoundError(f"Vector store not found at {self.db_name}")
        print(f"📂 Loading FAISS index from {self.db_name}")
        vector_store = FAISS.load_local(
            self.db_name,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded FAISS index with {len(vector_store.index_to_docstore_id)} documents.")
        return vector_store

    @staticmethod
    def format_docs(docs):
        """Combine retrieved documents into one text block."""
        with open("debug_docs.txt", "w") as f:
            f.write("\n\n".join([doc.page_content for doc in docs]))
        
        return "\n\n".join([doc.page_content for doc in docs])

    def _build_prompt_template(self):
        """Create chat prompt template."""
        # 1. Provide answers strictly based on the given context. If the answer is not available, respond with: "I’m sorry, I don’t have that information."

        return ChatPromptTemplate.from_template("""
You are a policy assistant for MOL Chemical Tankers. 
Answer precisely and only based on the provided context.

Guidelines:

1. Base your answer strictly on the provided context. Do not include unsupported information.

2. Determine responsibility based on specificity:
   2.1. If multiple categories or departments are mentioned, prioritize the most specific match between items in the question and those listed in the context.
   2.2. For example, if the question includes subtypes (e.g., "CLI/FDD") and the context mentions those subtypes under a specific department, that department takes priority.

3. Prefer specific over general:
   3.1. If the question refers to both a general and a specific category (e.g., P&I and FDD), select the department explicitly linked to the specific subtype.

4. Include key approval details:
   4.1. Extract Responsible Department / Head, Authorised Approvers, Deliberation/Report requirements, and Review.
   4.2. If general rules or special cases (e.g., "Important" vs "Other") apply, include them even if they are in a less specific section of the context.
   4.3. Prioritize specific matches first, then supplement with relevant general rules.

5. Handle multiple items separately:
   5.1. If the question involves multiple items or contracts with different amounts or types, list the approval criteria for each item individually before indicating the overall criteria.
   5.2. Clearly indicate: Responsible Department, Authorised Approvers, Deliberation/Report, and Review for each item.

6. Missing or unsupported information:
   6.1. If information is unavailable or not stated in the context, respond: "I’m sorry, I don’t have that information."

7. Style:
   7.1. Keep answers concise, factual, and professional.
   7.2. Use **narrative style per item**, like this example:

### 1. [Item Name] ([Amount])
- **Responsible Department:** [Department / Head]
- **Authorised Approvers:** [Approvers]
- **Deliberation/Report:** [Deliberation or report requirements, if any]
- **Review:** [Review department]

### 2. [Next Item] ([Amount])
- ...

- Include a short **Notes** section at the end if applicable:
  - If approvals are combined, specify which criteria apply (e.g., stricter / higher threshold).
  - Any rules about reporting thresholds or annual approvals.

---

Question: {question}  
Context: {context}  
Answer:
""")
    def _build_rag_chain(self):
        """Create the retrieval + generation chain."""
        prompt = self._build_prompt_template()
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, question: str):
        """Run the RAG chain for a given question."""
        print(f"💬 Asking: {question}")
        docs = self.retriever.invoke(question)
        response = self.rag_chain.invoke(question)
        # print("\n✅ Response:\n")
        return response, docs


# if __name__ == "__main__":
#     rag_bot = ThinkpalmRAG()
#     question = "In Thinkpalm, what is the definition of OFFICIAL AUTHORITY REGULATIONS ?"
#     answer = rag_bot.ask(question)
#     print(answer)
class ThinkpalmRagCosmos:
    """
Test RAG Retrieval + LLM using OpenAI GPT-4.1 and Azure Cosmos DB
Fixed vector search implementation
"""

    import os
    import logging
    import numpy as np
    from openai import OpenAI
    from azure.cosmos import CosmosClient, exceptions

    # ------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------
    COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
    COSMOS_KEY = os.getenv("COSMOS_KEY")
    COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
    COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    missing = [
        k for k, v in {
            "COSMOS_ENDPOINT": COSMOS_ENDPOINT,
            "COSMOS_KEY": COSMOS_KEY,
            "COSMOS_DATABASE": COSMOS_DATABASE,
            "COSMOS_CONTAINER": COSMOS_CONTAINER,
            "OPENAI_API_KEY": OPENAI_API_KEY
        }.items() if not v
    ]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

    # ------------------------------------------------------------
    # Initialize clients
    # ------------------------------------------------------------
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    container = cosmos_client.get_database_client(COSMOS_DATABASE).get_container_client(COSMOS_CONTAINER)

    # ------------------------------------------------------------
    # Search function — Azure Cosmos DB Vector Search
    # ------------------------------------------------------------
    def search_cosmos_documents_native(query_embedding, top_k=5):
        """
        Retrieve documents from Cosmos DB using native vector search.
        Uses VectorDistance function in SQL query for similarity search.
        """
        try:
            # Convert embedding to proper format
            embedding_str = str(query_embedding).replace(" ", "")
            
            # Build SQL query with VectorDistance function
            query_text = f"""
            SELECT TOP {top_k} 
                c.id, 
                c.text, 
                c.metadata,
                VectorDistance(c.vector_embedding, {embedding_str}) AS score
            FROM c
            ORDER BY VectorDistance(c.vector_embedding, {embedding_str})
            """

            results = container.query_items(
                query=query_text,
                enable_cross_partition_query=True
            )

            # Convert results to a list
            docs = []
            for doc in results:
                docs.append({
                    "id": doc.get("id"),
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": float(doc.get("score", 0))
                })

            return docs

        except Exception as e:
            logging.error(f"Native vector search error: {e}")
            return []


    # def search_cosmos_documents_manual(query_embedding, top_k=5):
    #     """
    #     Alternative: Manual cosine similarity calculation.
    #     Use this if VectorDistance is not available in your Cosmos DB tier.
    #     """
    #     try:
    #         # Fetch all documents (or use pagination for large datasets)
    #         query = "SELECT c.id, c.text, c.metadata, c.vector_embedding FROM c"
            
    #         results = container.query_items(
    #             query=query,
    #             enable_cross_partition_query=True
    #         )

    #         # Calculate cosine similarity manually
    #         scored_docs = []
    #         query_vec = np.array(query_embedding)
    #         query_norm = np.linalg.norm(query_vec)
            
    #         for doc in results:
    #             doc_vec = np.array(doc.get("vector_embedding", []))
    #             if len(doc_vec) == 0:
    #                 continue
                    
    #             # Cosine similarity
    #             doc_norm = np.linalg.norm(doc_vec)
    #             if doc_norm == 0 or query_norm == 0:
    #                 similarity = 0
    #             else:
    #                 similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
                
    #             scored_docs.append({
    #                 "id": doc.get("id"),
    #                 "text": doc.get("text", ""),
    #                 "metadata": doc.get("metadata", {}),
    #                 "score": float(similarity)
    #             })
            
    #         # Sort by score descending and take top_k
    #         scored_docs.sort(key=lambda x: x["score"], reverse=True)
    #         return scored_docs[:top_k]

    #     except Exception as e:
    #         logging.error(f"Manual vector search error: {e}")
    #         return []


    # ------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------
    PROMPT_TEMPLATE = """
    You are an assistant for MOLCT Management Approval guidance. 
    Answer ONLY using information from the retrieved context and previous conversation history.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # ------------------------------------------------------------
    # Generate GPT-4.1 response
    # ------------------------------------------------------------
    def generate_rag_response(query, retrieved_docs):
        if not retrieved_docs:
            return "This information is not found in the provided documents."

        context = "\n\n".join(
            [f"Document {i+1} (Score: {d['score']:.3f}):\n{d['text']}" for i, d in enumerate(retrieved_docs)]
        )

        final_prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI response error: {e}")
            return f"Error generating response: {e}"