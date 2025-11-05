import os
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import numpy as np

from dotenv import load_dotenv
load_dotenv()
import json
import random
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import OpenAI
from datetime import datetime
import os

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# memory.save_context({"input": "Hi"}, {"output": "Hello there!"})
# print(memory.load_memory_variables({}))

# exit()

from datetime import datetime
from typing import List, Dict
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import OpenAI
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
        """
        Format retrieved docs for LLM prompt — removes debug doc labels so model doesn’t cite “Doc X”.
        """
        return "\n\n".join(d.get("text", "") for d in docs)

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
    def _tokenize_search_hits(self, cand, docs):
        tok = cand.lower()
        hits = []
        for d in docs:
            if tok in d.get("text","").lower():
                hits.append(d)
        return hits
    
    def deduplicate_docs(self, retrieved_docs):
        seen_texts = set()
        unique_docs = []
        for d in retrieved_docs:
            text_norm = d.get("text", "").strip().lower()
            if text_norm not in seen_texts:
                seen_texts.add(text_norm)
                unique_docs.append(d)
        return unique_docs

    def ask(self, user_id, question):
        
        # --- Step 0: Classify the message (Using the efficient _is_small_talk) ---
        is_small_talk = self._is_small_talk(question)
        print("is_small_talk:", is_small_talk)
        
        if is_small_talk:
            # --- Small Talk Handling ---
            answer = self._generate_small_talk_response(question)

            # Update memory
            memory = self.get_memory(user_id)
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            
            return {"answer": answer, "related": False}
        
        # --- Step 1: Retrieve memory and rewrite query if needed ---
        memory = self.get_memory(user_id)
        past_context = memory.load_memory_variables({}).get("history", "")

        # --- Step 2: Extract canonical keyword (for better retrieval precision) ---
        keyword_to_check = self._extract_keywords(question)  # e.g., "Novation of the contract"
        print(f"[DEBUG] Extracted canonical keyword: {keyword_to_check}")

        # Build the retrieval query (including conversation history if any)
        if past_context:
            query_for_retrieval = self._rewrite_query(past_context, question)
        else:
            query_for_retrieval = question

        # Force include canonical keyword to improve recall of specific table rows
        if keyword_to_check:
            query_for_retrieval = f"{keyword_to_check} {query_for_retrieval}"
            print(f"[DEBUG] Forcing keyword into retrieval query: {keyword_to_check}")

        print(f"[DEBUG] Final Query for Retrieval: {query_for_retrieval}")

        # --- Step 3: Vector Retrieval ---
        # --- Step 3: Vector Retrieval (No change in top-level flow) ---
        # Embedding: use the OpenAI embeddings API properly and serialize
        emb = self.llm.embeddings.create(model="text-embedding-3-large", input=query_for_retrieval)
        query_embedding = emb.data[0].embedding

        # initial vector search (based on the rewritten query)
        retrieved_docs = self.search_cosmos_documents(query_embedding)

        # --- Step 4: Dynamic Keyword + Subtype Extraction & Robust Fallback ---
        # Use the LLM to extract the high-value canonical keyword (e.g., "Novation of Time Charter Contract")
        keyword_to_check = self._extract_keywords(question)  # existing method you have
        # New helper: extract specific subtype/entity (Time Charter, P&I, FDD, IT, etc.)
        subtype = None
        try:
            # Try a quick deterministic entity detection using regex/keywords first to avoid cost
            candidate_subtypes = ["time charter", "time charterer", "charter", "novation", "p&i", "fdd", "cli", "it", "it-related", "service agreement"]
            q_lower = (question + " " + (past_context or "")).lower()
            for cand in candidate_subtypes:
                if cand in q_lower:
                    subtype = cand
                    break
            # If no subtype found heuristically, ask the LLM for a single subtype label
            if not subtype:
                subtype_prompt = (
                    "Extract a single canonical subtype or domain-word from the question if present (e.g. 'Time Charter', 'P&I', 'IT', 'Service agreement'). "
                    "Return 'None' if no subtype is present.\n\nQuestion:\n" + question
                )
                resp = self.llm.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": subtype_prompt}],
                    temperature=0.0
                )
                subtype = resp.choices[0].message.content.strip().split('\n')[0]
                if subtype.lower() in {"none", ""}:
                    subtype = None
                else:
                    subtype = subtype
        except Exception as e:
            print(f"[DEBUG] subtype extraction error: {e}")
            subtype = None

        # Build a list of fallback keyword variations to try (ordered)
        fallback_candidates = []
        if keyword_to_check:
            keyword_clean = keyword_to_check.strip()
            fallback_candidates.append(keyword_clean)
            # simplified: remove filler words like 'of the', 'of'
            simplified = keyword_clean.replace(" of the ", " ").replace(" of ", " ").strip()
            if simplified and simplified != keyword_clean:
                fallback_candidates.append(simplified)
            # also try last token segments (e.g., "Time Charter Contract" -> "Time Charter")
            parts = keyword_clean.split()
            if len(parts) > 2:
                fallback_candidates.append(" ".join(parts[-3:]))  # last 3 words
                fallback_candidates.append(" ".join(parts[-2:]))  # last 2 words
        # Always prefer subtype if present
        if subtype:
            fallback_candidates.insert(0, subtype)

        # De-duplicate while preserving order
        seen_fc = set()
        fallback_candidates = [c for c in fallback_candidates if c and not (c in seen_fc or seen_fc.add(c))]

        # If we already did vector search, check if any candidate substring appears in retrieved docs
        joined_texts = " ".join([d.get("text", "").lower() for d in retrieved_docs])
        missing_candidates = [c for c in fallback_candidates if c and c.lower() not in joined_texts]

        # If any fallback candidates are missing, run progressive SQL fallback searches (CONTAINS)
        if missing_candidates:
            print(f"[DEBUG] Vector search did not include these candidates in top docs: {missing_candidates}")
            for cand in missing_candidates:
                # SQL safe-escape single quotes
                safe_kw = cand.replace("'", "''")
                # Use CONTAINS to match substrings (more relaxed than exact literal)
                fallback_query = f"""
                    SELECT TOP 20 c.id, c.text, c.metadata
                    FROM c
                    WHERE CONTAINS(LOWER(c.text), '{safe_kw.lower()}')
                """
                try:
                    fallback_results = list(self.container.query_items(query=fallback_query, enable_cross_partition_query=True))
                except Exception as e:
                    print(f"[DEBUG] fallback SQL for '{cand}' error: {e}")
                    fallback_results = []
                # Add non-duplicate docs to retrieved_docs (preserve earlier docs first)
                existing_ids = {d['id'] for d in retrieved_docs}
                added = 0
                for d in fallback_results:
                    if d.get("id") not in existing_ids:
                        retrieved_docs.append({"id": d.get("id"), "text": d.get("text", ""), "metadata": d.get("metadata", {}), "score": 0.0})
                        existing_ids.add(d.get("id"))
                        added += 1
                print(f"[DEBUG] Fallback search for '{cand}' added {added} docs")

            # If still zero matches for all fallback candidates, try a relaxed substring search on keyword tokens
            if all(len(self._tokenize_search_hits(cand, retrieved_docs)) == 0 for cand in fallback_candidates):
                print("[DEBUG] No fallback docs found with CONTAINS; trying relaxed token search on key tokens.")
                tokens = []
                for cand in fallback_candidates:
                    tokens += [t for t in cand.split() if len(t) > 2]
                tokens = list(dict.fromkeys(tokens))[:6]  # unique tokens, limit to 6
                for tok in tokens:
                    safe_tok = tok.replace("'", "''").lower()
                    token_query = f"""
                        SELECT TOP 10 c.id, c.text, c.metadata
                        FROM c
                        WHERE CONTAINS(LOWER(c.text), '{safe_tok}')
                    """
                    try:
                        token_res = list(self.container.query_items(query=token_query, enable_cross_partition_query=True))
                    except Exception as e:
                        token_res = []
                    existing_ids = {d['id'] for d in retrieved_docs}
                    for d in token_res:
                        if d.get("id") not in existing_ids:
                            retrieved_docs.append({"id": d.get("id"), "text": d.get("text", ""), "metadata": d.get("metadata", {}), "score": 0.0})
                            existing_ids.add(d.get("id"))


        retrieved_docs = self.deduplicate_docs(retrieved_docs)
        # --- Step 5: Preparing Context (Improved debug block) ---
        context = self.format_context(retrieved_docs)
        print(f"[DEBUG] Retrieved docs count (post-fallback merge): {len(retrieved_docs)}")

        # Print top few retrieved document snippets for verification
        for i, d in enumerate(retrieved_docs[:6]):
            preview = d.get("text", "").replace("\n", " ").strip()[:180]
            print(f"[DEBUG] Doc {i+1} | Score: {d.get('score', 0):.3f} | Snippet: {preview}...")

        # Save full retrieval trace for external inspection
        with open("Verification_retrieved_docs.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n==============================\n")
            f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Rewritten Query: {query_for_retrieval}\n")
            f.write(f"Extracted Keyword: {keyword_to_check}\n")
            f.write(f"Extracted Subtype: {subtype}\n")
            f.write(f"Final Fallback Candidates: {fallback_candidates}\n")
            f.write(f"Retrieved Docs: {len(retrieved_docs)}\n\n")
            for i, d in enumerate(retrieved_docs):
                f.write(f"Doc {i+1} (Score {d.get('score', 0):.3f}):\n{d.get('text','')}\n\n")
            f.write("==============================\n")
            
        # --- Step 4: Generate Final Answer ---
        # The final prompt includes everything: the original question, memory, and context.
        # We now trust the memory + LLM to use the best information.
        # You must **only extract lines directly related to the user’s topic** (for example, if the user asks about *product acquisition*, do not include other categories like *claim settlement* or *consultancy agreements*). 
# Do not summarize or interpret — copy the exact relevant block only.

        prompt = f"""
        You are **Thinkpalm’s Corporate Knowledge Assistant**. Your task is to generate a **precise, complete, and policy-accurate** answer to the user’s question using **only** the information in the 'Document Context' and related policy tables.

        ==============================
        ### CORE INSTRUCTIONS
        ==============================

        1. **Answer directly and decisively** — begin with a clear conclusion that addresses the user’s question (do not restate the question).

        2. **Preserve every essential policy detail** from the Document Context:
        - Always include: **Authorised Approver(s)**, **Co-Management Department(s)** (if any), **Deliberation**, **Report**, **Review**, and **CC** if they exist.  
        - Use the **exact tokens** from the document (e.g., “A1”, “BDM”, “MM”, “Email”, “GPM”, “AF”, “BS”).  
        - Do **not** summarize, rename, or paraphrase them.

        3. **Integrate related context cohesively** — when multiple clauses or table rows refer to the same action (e.g., “Novation”, “Amendment”, “Cancellation”), merge their logic into a single, coherent answer.

        4. **Handle exceptions and nuances explicitly.**  
        Example: include phrases like “subject to GPM review”, “requires MM email report”, or “HOD to determine importance” exactly as stated.

        5. **No filler or speculation.**  
        Only return factual statements that can be traced to the Document Context.

        ==============================
        ### NOVATION / AMENDMENT / CANCELLATION RULES
        ==============================

        6. If the question relates to **Novation**, **Amendment**, or **Cancellation**:
        - Identify and use the *exact policy title* from the Document Context.  
        - Include all associated approvals, deliberations, and reviews.  
        - If the clause includes department co-management (e.g., AF, BS) or duration-based rules (e.g., “5 years or more”), list them precisely.
        - If stated, include “GPM HOD decides whether it is Important or Others” **verbatim**.

        ==============================
        ### DEPARTMENT & SUBTYPE LOGIC
        ==============================

        7. If the question or context mentions a **subtype** (e.g., CLI, FDD, DTH, TCL):
        - Identify and include the **specific department** responsible for that subtype, even if another department handles broader or related categories.  
        - Example mappings:
            - “FDD”, “TCL”, “DTH” → Business Planning Department  
            - “P&I (General)” → Ship Management Department  

        8. When multiple departments appear:
        - Apply the **most specific rule** (the subtype’s department takes precedence).  
        - Mention both departments only if the Document Context shows overlapping responsibilities.

        ==============================
        ### COST-RELATED DECISIONS
        ==============================

        9. For multi-cost or multi-item questions, follow these rules:

            a. **Treat each category separately:** For each transaction category (e.g., Implementation, Maintenance), sum all relevant amounts across sub-allocations (MCT, UNIX, subsidiaries, etc.) to determine the total cost for that category. Do NOT separate by internal allocations.

            b. **Identify Categories:** Determine the distinct approval categories and their specific thresholds based on the total category amount.
                * Implementation / New System → "Acquisition, disposal of IT related fixed assets"
                * Maintenance / Service Contract → "IT-related service agreements"

            c. **Extract All Details:** For each category, extract the entire line of approval details (Approvers, Reports, Reviews, Co-management, etc.) from the Document Context that matches the applicable threshold.

            d. **Final Rule Application:** State the final business rule that governs the submission based on total category amounts.

            e. **Output Format for Multiple Categories:**
                A) For [First Transaction Category] — [total amount + approval details].  
                B) For [Second Transaction Category] — [total amount + approval details].  
                Then add a short **“Conclusion”** explaining the overall rule.  
        ==============================
        ### CONTEXTUAL FILTERING RULES
        ==============================

        10. If the question explicitly mentions a **policy year** (e.g., “Policy Year 2025”):
            - Interpret it as referring to the **annual plan** for that policy year.  
            - Only include approval criteria applicable to the annual plan.  
            - **Exclude unrelated categories** like “Important” or “Others” unless explicitly required by the question.

        11. If a **policy year** is not specified, apply general approval criteria relevant to the subject matter.

        ==============================
        ### STRUCTURE & OUTPUT STYLE
        ==============================

        12. Structure your answer as follows:
        - **Opening Summary:** One clear sentence answering the question directly.  
        - **Detailed Breakdown:** Use numbered or bulleted structure to present key facts (departments, approvers, thresholds, deliberation/review, etc.).  
        - **Conclusion:** End with a decisive sentence summarizing the applicable rule or final action required.

        13. Maintain a professional, factual tone.  
            Avoid duplication, long-winded explanations, or internal references like “Doc 1” or “Page 17”.

        ==============================
        ### INPUT CONTEXT
        ==============================

        Conversation History:
        {past_context}

        Document Context:
        {context}

        User Question:
        {question}

        ==============================
        ### OUTPUT
        ==============================

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
        with open("Verification_retrieved_docs.txt", "a") as f:
            f.write(f"\n\nAnswer\n==========================\n{answer}\n\n")
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
