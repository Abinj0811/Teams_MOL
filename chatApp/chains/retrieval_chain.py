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
import json
import re
import random



from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import OpenAI
from datetime import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# memory.save_context({"input": "Hi"}, {"output": "Hello there!"})
# print(memory.load_memory_variables({}))

# exit()

from datetime import datetime
from typing import List, Dict
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import OpenAI
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
        
        self.SMALL_TALK_RESPONSES = {
        "greeting": [
            "Hello! I am Thinkpalm's Corporate Knowledge Assistant. How may I be of assistance with your business query?",
            "Good day. Thank you for reaching out. I'm ready to help with any policy or knowledge questions you may have.",
            "Hi there. I trust you are having a productive day. Please let me know your question.",
            "Welcome! I am here to provide accurate and professional support. What information are you seeking?",
        ],
        "thanks": [
            "You are most welcome. Is there anything else I can clarify or retrieve for you?",
            "My pleasure. Do not hesitate to ask if further information is required.",
            "Glad to be of assistance. Have a productive day.",
        ],
        "who_are_you": [
            "I am Thinkpalm's Corporate Knowledge Assistant, designed to provide information and policy details from our internal knowledge base.",
        ],
        "generic_positive": [
            "That is kind of you to say. I am functioning optimally and ready to address your corporate queries.",
        ]
    }

    # ------------------------------------------------------------
    # MEMORY MANAGEMENT
    # ------------------------------------------------------------
    def get_memory(self, user_id):
        """Create or retrieve summary memory per user."""
        if user_id not in self.memory_store:
            self.memory_store[user_id] = ConversationSummaryMemory(
                llm=ChatOpenAI(model="gpt-4.1", temperature=0),
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
        """
        Query Cosmos using the VectorDistance API. Ensure the embedding is JSON-serialized
        so Cosmos receives a correct array. Returns a list of docs with id, text, metadata, score.
        """
        # Use json.dumps to produce a properly formatted array literal
        embedding_str = json.dumps(query_embedding)

        query = f"""
        SELECT TOP {self.top_k}
            c.id, c.text, c.metadata,
            VectorDistance(c.vector_embedding, {embedding_str}) AS score
        FROM c
        ORDER BY VectorDistance(c.vector_embedding, {embedding_str})
        """
        try:
            results = list(self.container.query_items(query=query, enable_cross_partition_query=True))
        except Exception as e:
            print(f"Error running vector query: {e}")
            results = []

        docs = []
        for d in results:
            docs.append({
                "id": d.get("id"),
                "text": d.get("text", ""),
                "metadata": d.get("metadata", {}),
                "score": float(d.get("score", 0))
            })
        # debug
        print(f"[DEBUG] search_cosmos_documents returned {len(docs)} docs (top_k={self.top_k})")
        return docs

    def _keyword_fallback_search(self, keyword):

        """

        If vector retrieval misses the exact table row, use a simple keyword substring search

        against the stored documents in Cosmos (or text index). This helps guarantee we get

        explicit lines like "Appointment/ Removal of Directors/ EOs".

        """

        # escape single quotes in keyword for SQL

        safe_kw = keyword.replace("'", "''")

        query = f"""

        SELECT TOP 20 c.id, c.text, c.metadata

        FROM c

        WHERE CONTAINS(c.text, '{safe_kw}')

        """

        try:

            results = list(self.container.query_items(query=query, enable_cross_partition_query=True))

        except Exception as e:

            print(f"Keyword fallback search error: {e}")

            results = []



        docs = []

        for d in results:

            docs.append({

                "id": d.get("id"),

                "text": d.get("text", ""),

                "metadata": d.get("metadata", {}),

                "score": 0.0

            })

        print(f"[DEBUG] _keyword_fallback_search('{keyword}') found {len(docs)} docs")

        return docs

    @staticmethod
    def format_context(docs: List[Dict]) -> str:
        # Return the raw text blocks joined so the LLM sees the exact table rows (no truncation).
        return "\n\n".join([f"Doc {i+1} (Score {d.get('score', 0):.3f}):\n{d.get('text','')}" for i, d in enumerate(docs)])


    # Add this new helper method to your class (or implement the logic inline)
    def _rewrite_query(self, memory_history: str, current_question: str) -> str:
        """Uses the LLM to convert an ambiguous follow-up into a standalone query."""
        
        # This prompt instructs the LLM to perform the rewriting task.
        rewrite_prompt = f"""
                You are a query rewriter for a Retrieval-Augmented Generation (RAG) system.

        Your goal is to rewrite the 'Current User Question' into a single, self-contained,
        and contextually complete search query using the 'Conversation History' to fill in
        missing references.

        Strict Rules:
        1. Preserve the logical meaning, phrasing, and operators (e.g., “and”, “or”, “/”) exactly as in the user’s question. 
        - Do NOT replace “or” with “and”, or vice versa.
        - Do NOT merge multiple conditions unless they are identical in meaning.
        2. Do NOT infer or generalize beyond the user’s wording — stay faithful to the intent.
        3. Include relevant context from Conversation History only if it clarifies **what** the user is referring to.
        4. Output must be a single concise query — no explanations, no extra text.

        Example:
        History:
        User: What are the benefits of the new HR policy?
        Assistant: The policy provides flexible PTO and a stipend.
        Current User Question: What is the stipend amount?
        Rewritten Query: What is the stipend amount for the new HR policy?

        Conversation History:
        {memory_history}

        Current User Question:
        {current_question}
        Rewritten Query:
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4.1", # Use a fast, inexpensive model for this step
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0 # Set low temperature for factual rewriting
        )
        return response.choices[0].message.content.strip()
    # ------------------------------------------------------------
    # ASK METHOD (MAIN PIPELINE)
    # ------------------------------------------------------------
    # New (Simplified) ask method structure:
    def _extract_keywords(self, question: str) -> str | None:
        """Uses the LLM to extract the exact, canonical policy title for fallback search."""
        
        # 🚨 KEY CHANGE: The prompt explicitly asks for the 'EXACT POLICY TITLE'
        extraction_prompt = f"""
        You are a policy title extractor. Your task is to identify the single, canonical, and **EXACT POLICY TITLE** corresponding to the user's question, which must be used for a literal database lookup.
        
        The extracted title MUST be a complete, literal match for the policy in question.
        
        If the question is general (e.g., 'What is the cost?', 'How are you?'), return 'None'.

        Examples (Use these as templates for mapping user intent to exact title):
        - User: Who approves the Appointment of new directors? -> Output: Appointment/ Removal of Directors/ EOs
        - User: What is the process for paying a cancellation fee? -> Output: Payment of cancellation fee/ penalty charge
        - User: What is the travel policy? -> Output: Business Travel
        - User: What is the cost? -> Output: None
        
        User Question:
        {question}
        
        Extracted Keyword (or 'None'):
        """
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.0
            )
            keyword = response.choices[0].message.content.strip()
            # Cleaning the output
            keyword = keyword.replace('"', '').replace("'", '').split('\n')[0].strip()
            return keyword if keyword.lower() != 'none' and keyword else None
        except Exception as e:
            # Handle exceptions gracefully
            return None

    
# Define these as class attributes in your Knowledge Assistant class
# or as constants accessible by the method.
    
    def _is_small_talk(self, question: str) -> bool:
        """
        Classifies a question as small talk using deterministic rules.
        This version is strict, requiring the small talk phrase to dominate the query.
        """
        
        question_lower = question.lower().strip()
        question_words = question_lower.split()
        
        # 1. Define common small talk keywords
        GREETINGS = ["hello",'hloo', "hi", "hey", "good morning", "good evening", "greetings"]
        INQUIRIES = ["how are you", "what's up", "what are you doing", "who are you"]
        AFFIRMATIONS = ["thank you", "thanks", "i appreciate it", "bye", "goodbye"]
        
        # Consolidate phrases, including squashed versions (e.g., "thankyou")
        all_phrases = GREETINGS + AFFIRMATIONS + INQUIRIES
        squashed_phrases = [p.replace(' ', '').replace("'", "") for p in all_phrases if ' ' in p]
        all_checks = all_phrases + squashed_phrases
        
        # 2. Iteratively check for substring matches with strict dominance rules
        for phrase in all_checks:
            if phrase in question_lower:
                phrase_len = len(phrase.split())
                question_len = len(question_words)
                
                # A. Exact Match (The most definitive check)
                if phrase == question_lower:
                    return True
                
                # B. Dominance Check: The small talk phrase is nearly the entire query (e.g., 1-2 extra words)
                # Example: "Hi there" (2 words) or "Thank you so much" (4 words)
                if question_len <= phrase_len + 2:
                    return True

                # C. Boundary Check for leading/trailing small talk
                # This catches "Hi, can you tell me the policy?" only if the policy part is also short (max 5 words)
                if (question_lower.startswith(phrase) or question_lower.endswith(phrase)) and question_len <= 5:
                    return True
                    
        # 3. Heuristic: Final catch for very short, non-standard simple queries (e.g., "Thanks")
        if len(question_words) <= 2 and any(word in question_lower for word in GREETINGS + ["thanks", "bye"]):
            return True
        
        return False
    def _generate_small_talk_response(self, question: str) -> str:
        """
        Selects a deterministic, professional response based on question type.
        """
        question_lower = question.lower().strip()
        
        # Check what kind of small talk it is (based on the same logic as _is_small_talk)
        if any(phrase in question_lower for phrase in ["hello", "hi", "hey", "good morning", "greetings"]):
            key = "greeting"
        elif any(phrase in question_lower for phrase in ["thank you", "thanks", "i appreciate it"]):
            key = "thanks"
        elif any(phrase in question_lower for phrase in ["who are you", "what is your name"]):
            key = "who_are_you"
        else:
            # Default for other simple small talk (like "how are you")
            key = "generic_positive" 

        # Select a random response from the category
        return random.choice(self.SMALL_TALK_RESPONSES.get(key, self.SMALL_TALK_RESPONSES['greeting']))

    def ask(self, user_id, question):
        
        # --- Step 0: Classify the message (Using the efficient _is_small_talk) ---
        is_small_talk = self._is_small_talk(question)
        print("is_small_talk:", is_small_talk)
        # return
        
        if is_small_talk:
            # --- Small Talk Handling (REPLACED LLM RESPONSE GENERATION) ---
            
            # Generate the response using the deterministic function
            answer = self._generate_small_talk_response(question)
            
            # Update memory
            memory = self.get_memory(user_id)
            # memory.chat_memory.add_user_message(question)
            # memory.chat_memory.add_ai_message(answer)
            
            return {"answer": answer, "related": False}
        
        # --- Step 1 & 2: Get History and Rewrite Query (No change) ---
        memory = self.get_memory(user_id)
        past_context = memory.load_memory_variables({}).get("history", "")
        if not past_context:
            query_for_retrieval = question
        else:
            query_for_retrieval = self._rewrite_query(past_context, question)

        print(f"[DEBUG] Rewritten Query for Retrieval: {query_for_retrieval}")

        # --- Step 3: Vector Retrieval (No change) ---
        # Embedding: use the OpenAI embeddings API properly and serialize
        emb = self.llm.embeddings.create(model="text-embedding-3-large", input=query_for_retrieval)
        query_embedding = emb.data[0].embedding

        retrieved_docs = self.search_cosmos_documents(query_embedding)

        # --- Step 4: Dynamic Keyword Fallback (REPLACED HARDCODING) ---
        
        # Use the LLM to dynamically get the high-value keyword (e.g., "Appointment/ Removal of Directors")
        keyword_to_check = self._extract_keywords(question) 

        if keyword_to_check:
            joined_texts = " ".join([d.get("text", "") for d in retrieved_docs]).lower()
            
            # Check if the dynamic keyword is present in the current vector search results (to avoid fallback if successful)
            if keyword_to_check.lower() not in joined_texts:
                print(f"[DEBUG] Vector search missed dynamic keyword: '{keyword_to_check}'. Running SQL fallback.")
                
                # try a direct keyword search using the dynamically extracted phrase
                fallback_docs = self._keyword_fallback_search(keyword_to_check)
                
                # merge (while keeping previously found docs first)
                # avoid duplicates by id
                ids = {d['id'] for d in retrieved_docs}
                for d in fallback_docs:
                    if d['id'] not in ids:
                        retrieved_docs.append(d)
            else:
                print(f"[DEBUG] Vector search successfully captured dynamic keyword: '{keyword_to_check}'. Skipping fallback.")

        # --- Step 5: Preparing Context (Modified to include new debug line) ---
        context = self.format_context(retrieved_docs)
        print(f"[DEBUG] Retrieved docs count: {len(retrieved_docs)}")
        
        # optional: save raw context for debugging
        with open("retrieved_docs.txt", "a") as f:
            f.write(f"\n\nQuestion: {query_for_retrieval}\nContext:\n{context}\n")
        
        
        # --- Step 4: Generate Final Answer ---
        # The final prompt includes everything: the original question, memory, and context.
        # We now trust the memory + LLM to use the best information.
        # You must **only extract lines directly related to the user’s topic** (for example, if the user asks about *product acquisition*, do not include other categories like *claim settlement* or *consultancy agreements*). 
# Do not summarize or interpret — copy the exact relevant block only.

        prompt = f"""
        Your task is to act as Thinkpalm’s Corporate Knowledge Assistant. Your goal is to provide a single, comprehensive answer to the user's question by synthesizing all relevant facts from the provided 'Document Context' and related data tables.

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
10) From the provided Document Context, IDENTIFY the exact Authorised Approver tokens and Reviewers for the given User Question.
You MUST NOT paraphrase them; return them EXACTLY as they appear in the Document Context (for example: MOL, BDM, A1, A2, GPM, etc).

11)Instruction for multiple costs:

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
            model="gpt-4.1",
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
