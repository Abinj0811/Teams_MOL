import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# # Depending on your chosen LLM:
# from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()



from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import OpenAI
from datetime import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity

from langchain_classic.memory import ConversationBufferMemory

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# memory.save_context({"input": "Hi"}, {"output": "Hello there!"})
# print(memory.load_memory_variables({}))

# exit()

from datetime import datetime
from typing import List, Dict
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_classic.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI


class ThinkpalmCosmosRAGmethod2:
    def __init__(self, cosmos_endpoint, cosmos_key, db_name, container_name, history_container_name):
        # ⚠️ Move to env var later

        # Cosmos setup
        # print(cosmos_endpoint, cosmos_key)

        self.client = CosmosClient(url=cosmos_endpoint, credential=cosmos_key)
        self.db = self.client.get_database_client(db_name)
        self.container = self.db.get_container_client(container_name)
        self.history_container = self.db.get_container_client(history_container_name)
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # OpenAI API client
        self.llm = OpenAI(api_key=OPENAI_API_KEY)

        # In-memory chat memory store (per user)
        self.memory_store = {}
        self.top_k = 5
        self.last_question_cache = {}

    # ------------------------------------------------------------
    # MEMORY MANAGEMENT
    # ------------------------------------------------------------
    def get_memory(self, user_id):
        """Create or retrieve summary memory per user."""
        if user_id not in self.memory_store:
            self.memory_store[user_id] = ConversationSummaryMemory(
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                return_messages=True
            )
        return self.memory_store[user_id]

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
    # CHAT HISTORY STORAGE (COSMOS)
    # ------------------------------------------------------------
    def get_user_history(self, user_id):
        try:
            query = f"""
            SELECT * FROM c 
            WHERE c.user_id = '{user_id}' 
            ORDER BY c.timestamp DESC OFFSET 0 LIMIT 5
            """
            items = list(self.history_container.query_items(query=query, enable_cross_partition_query=True))
            items.reverse()
            return items
        except exceptions.CosmosResourceNotFoundError:
            print("⚠️ Chat history container not found. Creating now.")
            self.history_container = self._ensure_container("chathistory", partition_key="user_id")
            return []
        except Exception as e:
            print(f"Error reading history: {e}")
            return []

    def save_chat_message(self, user_id, user_msg, assistant_msg):
        item = {
            "id": f"{user_id}-{datetime.utcnow().isoformat()}",
            "user_id": user_id,
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            self.history_container.upsert_item(item)
        except Exception as e:
            print(f"Error saving message: {e}")

    # ------------------------------------------------------------
    # RAG SEARCH
    # ------------------------------------------------------------
    def search_cosmos_documents(self, query_embedding):
        embedding_str = str(query_embedding).replace(" ", "")
        query = f"""
        SELECT TOP {self.top_k}
            c.id, c.text, c.metadata,
            VectorDistance(c.vector_embedding, {embedding_str}) AS score
        FROM c
        ORDER BY VectorDistance(c.vector_embedding, {embedding_str})
        """
        results = self.container.query_items(query=query, enable_cross_partition_query=True)
        docs = []
        for d in results:
            docs.append({
                "id": d.get("id"),
                "text": d.get("text", ""),
                "metadata": d.get("metadata", {}),
                "score": float(d.get("score", 0))
            })
        return docs

    @staticmethod
    def format_context(docs: List[Dict]) -> str:
        return "\n\n".join(
            [f"Doc {i+1} (Score {d['score']:.3f}):\n{d['text']}" for i, d in enumerate(docs)]
        )

    # Add this new helper method to your class (or implement the logic inline)
    def _rewrite_query(self, memory_history: str, current_question: str) -> str:
        """Uses the LLM to convert an ambiguous follow-up into a standalone query."""
        
        # This prompt instructs the LLM to perform the rewriting task.
        rewrite_prompt = f"""
        You are a query rewriter for a Retrieval-Augmented Generation (RAG) system. 
        Your task is to rewrite the 'Current User Question' into a single, comprehensive, 
        standalone search query based on the 'Conversation History'. The output must be 
        only the rewritten query and nothing else.

        Example:
        History: User: What are the benefits of the new HR policy? Assistant: The policy provides flexible PTO and a stipend.
        Current User Question: What is the stipend amount?
        Rewritten Query: What is the stipend amount for the new HR policy?
        
        Conversation History:
        {memory_history}
        
        Current User Question:
        {current_question}
        
        Rewritten Query:
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini", # Use a fast, inexpensive model for this step
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0 # Set low temperature for factual rewriting
        )
        return response.choices[0].message.content.strip()
    # ------------------------------------------------------------
    # ASK METHOD (MAIN PIPELINE)
    # ------------------------------------------------------------
    # New (Simplified) ask method structure:

    def ask(self, user_id, question):
        # --- Step 0: Classify the message ---
        intent = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify if the message is small talk or a business query."},
                {"role": "user", "content": question}
            ]
        )
        classification = intent.choices[0].message.content.lower()
        
        if "small talk" in classification:
            # Let the LLM respond naturally, but in the corporate knowledge assistant tone
            small_talk_response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are Thinkpalm's Corporate Knowledge Assistant. "
                        "Even when replying to casual greetings or small talk, "
                        "maintain a professional, courteous, and helpful tone. "
                        "Respond in a friendly but corporate style."
                    )},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            answer = small_talk_response.choices[0].message.content.strip()
            
            # Update memory
            memory = self.get_memory(user_id)
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            
            return {"answer": answer, "related": False}
    
        # --- Step 1 & 2: Get History and Rewrite Query ---
        memory = self.get_memory(user_id)
        past_context = memory.load_memory_variables({}).get("history", "")
        if not past_context:
            # If no history, the original question is the search query
            query_for_retrieval = question
        else:
            # LLM rewrites the query for optimal semantic search
            query_for_retrieval = self._rewrite_query(past_context, question)
            
        print(f"Rewritten Query for Retrieval: {query_for_retrieval}") # Helpful debug
        
        # --- Step 3: Embed and Retrieve Documents (RAG) ---
        emb = self.llm.embeddings.create(model="text-embedding-3-large", input=query_for_retrieval)
        query_embedding = emb.data[0].embedding
        
        retrieved_docs = self.search_cosmos_documents(query_embedding)
        context = self.format_context(retrieved_docs)
        
        with open("retrieved_docs.txt", "a") as f: 
            f.write(f"\n\nQuestion: {query_for_retrieval}\n") 
            f.write(f"Context:\n{context}\n")
        
        # --- Step 4: Generate Final Answer ---
        # The final prompt includes everything: the original question, memory, and context.
        # We now trust the memory + LLM to use the best information.
        # You must **only extract lines directly related to the user’s topic** (for example, if the user asks about *product acquisition*, do not include other categories like *claim settlement* or *consultancy agreements*). 
# Do not summarize or interpret — copy the exact relevant block only.

        prompt = f"""
        Your task is to act as Thinkpalm’s Corporate Knowledge Assistant. Your goal is to provide a single, comprehensive answer to the user's question by synthesizing all relevant facts from the provided 'Document Context' and related data tables.

IMPORTANT: Explicitly list all Authorised Approvers exactly as mentioned in the context (e.g., MOL, BDM, A1). Do NOT omit any approver, even if it seems implied. Ensure the final answer fully reflects the documents.

Instructions:

1) Directly addresses the user’s question and provides a clear conclusion.

2) Includes all relevant supporting details from the provided context — do not omit important nuances, exceptions, or conditions and important details such as authorisers,review , reporting, department or management (if mentioned )

3) Avoids copying the context verbatim — instead, paraphrase fluently while preserving the meaning.

4) Keeps the explanation concise and logically ordered, but not at the cost of losing essential information.

5) When multiple context points relate to the question, integrate them cohesively (don’t list separately).

6) If there are exceptions, authority levels, or conditions, explicitly mention them.

7) End with a final conclusion sentence that answers the question clearly and decisively.
8) If the question or context mentions a subtype (e.g., CLI/FDD, DTH, TCL, etc.),
   identify and include the department or rule specifically associated with that subtype,
   even if another department handles related general cases.
9) When multiple departments are involved:
   - Determine responsibility based on the **most specific match** (e.g., FDD → Business Planning Dept.).
   - Mention all relevant departments **only if their responsibilities overlap**.
10)Instruction for multiple costs:

    1.  **Disaggregate Costs:** Do NOT aggregate the costs. Treat each item (New System: US$58,000 and Maintenance Contract: US$38,000) as a separate transaction for the purpose of finding its individual criteria.
    2.  **Identify Categories:** Determine the two distinct approval categories and their specific thresholds:
        * US$58,000 for a new system is classified as **"Acquisition, disposal of IT related fixed assets."**
        * US$38,000 for a maintenance service contract is classified as **"IT-related service agreements."**
    3.  **Extract All Details:** For each transaction, extract the entire line of approval details (Approvers, Reports, Reviews, Co-management, etc.) from the Document Context that matches the applicable threshold.
    4.  **Final Rule Application:** State the final business rule that governs the submission when two different criteria apply.


    5. Output Format for multiple costs:compose the answer using this two-part structure, followed by the final decision rule. The output must be concise and based **ONLY** on the provided Document Context.

        A) For the [First Transaction Category]...** (State the applicable threshold and full criteria extracted from the context.)

        B) For the [Second Transaction Category]...** (State the applicable threshold and full criteria extracted from the context.)

        
        Conversation History:
        {past_context}
        
        Document Context:
        {context}
        
        User Question:
        {question} # Use the ORIGINAL question here, as the final answer must address it.
        
        Answer:
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        
        # --- Step 5: Update Memory and Cache ---
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        self.save_chat_message(user_id, question, answer)
        # self.last_question_cache[user_id] = question # <<< Remove this line, as it's no longer needed
        
        return {"answer": answer, "related": bool(past_context)} # 'related' is now simply based on whether a history exists
        

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
# """
if __name__ == "__main__":
    # COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
    # COSMOS_KEY = os.getenv("COSMOS_KEY")
    # DB_NAME = "thinkpalm_db"
    # DOCS_CONTAINER = "ragEmbedding"
    
    COSMOS_ENDPOINT=os.getenv("COSMOS_ENDPOINT")

    COSMOS_KEY=os.getenv("COSMOS_KEY")

    COSMOS_DATABASE=os.getenv("COSMOS_DATABASE")

    COSMOS_CONTAINER=os.getenv("COSMOS_CONTAINER")
        
        
    CHAT_CONTAINER = os.getenv("CHAT_CONTAINER")
    # print(COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER, CHAT_CONTAINER)
    # exit()
    rag = ThinkpalmCosmosRAGmethod2(COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER, CHAT_CONTAINER)

    # answer = rag.ask("user_id", "hi")
    # print(f"Assistant: {answer}\n")
    # user_id = input("Enter user_id: ")
    user_id= "test_user"

    print("\nType 'exit' to stop chatting.\n")
    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ["exit", "quit"]:
            print("Ending chat...")
            break
        response = rag.ask(user_id, user_msg)
        ans, related = response["answer"], response["related"]
        print(f"Assistant: {ans}\n, {related}\n")
        # 
# """
