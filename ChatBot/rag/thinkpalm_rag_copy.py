# rag/thinkpalm_rag.py
# Full ThinkpalmRAG class reconstructed from your uploaded chat_node.py
# Source: user-uploaded file chat_node.py. :contentReference[oaicite:1]{index=1}

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime
from tempfile import TemporaryDirectory
import json
import random
import os
import csv

from dotenv import load_dotenv

# NLP
import spacy

# Cosmos DB
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# LangChain / OpenAI wrappers (as in your original code)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")
load_dotenv()

# Logger
logger = logging.getLogger("ThinkpalmRAG")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def extract_significant_words(sentence: str):
    doc = nlp(sentence)
    return [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "NUM"} and not token.is_stop
    ]


class ThinkpalmRAG:
    def __init__(self):
        # ========== CONFIG ==========
        self.cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("COSMOS_KEY")
        self.db_name = os.getenv("COSMOS_DATABASE")
        self.container_name = os.getenv("COSMOS_CONTAINER")
        self.history_container_name = os.getenv("CHAT_CONTAINER")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "text-embedding-3-large")

        self.top_k = 5
        self.chat_memory = {}  # { user_id: deque([(user_msg, assistant_msg), ...]) }
        self.last_sync_counter = {}   # track per-user unsynced turns
        self.history_limit = 5
        self.autosave_interval = 12
        self.last_persist_index = defaultdict(int)
        self._last_rewritten = {
    "rewritten": None,
    "original": None,
    "history_used": None
}
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
        # RULE COMPONENTS
        self.COMMITTEE_RULES = """
            If the question involves a committee:
            - Identify and list the committee structure if available in context.
            - Copy names and roles verbatim; omit sections not found.
            """

        self.COST_RULES = """
            9. If the question concerns cost, budget, or amount:
                - Determine approval thresholds by total category amount.
                - Do not merge unrelated categories.     
                - Identify Categories : Eg:        
                - For multi-cost or multi-item questions, follow these rules:
                    a. **Identify Categories:** Determine the distinct approval categories and their specific thresholds based on the total category amount from the respective session.
                        Eg: * Implementation / New System → "Acquisition, disposal of IT related fixed assets"
                                policies under this session
                            * Maintenance / Service Contract → "IT-related service agreements"
                                policies under this session

                    c. **Extract All Details:** For each category, extract the entire line of approval details (Approvers, Reports, Reviews, Co-management, CC etc.) from the Document Context that matches the applicable threshold.

                    d. **Final Rule Application:** State the final business rule that governs the submission based on total category amounts.

                    e. **Output Format for Multiple Categories:**
                        A) For [First Transaction Category] — [total amount + approval details].  
                        B) For [Second Transaction Category] — [total amount + approval details].  
                        Then add a short **“Conclusion”** explaining the overall rule.
            """

        self.DISAMBIGUATION_RULE = """
        CATEGORY SELECTION & MERGE RULE:
        - If the question explicitly mentions 'IT', 'information technology', 'software', 'DXS', or 'ICS', use **only the IT-related assets** section.
        - If the question explicitly says 'Excluding IT' or 'Non-IT', use **only the Excluding IT-related assets** section.
        - If the question does NOT mention either IT or Non-IT, you must:
            1. Identify both **IT-related** and **Excluding IT-related** sections in the context.
            2. Select the relevant threshold line (e.g., 'Less than US$50,000') separately within each section.
            3. Present both results distinctly in the output, using format:

            **For IT-related assets:**  
            - [threshold line]

            **For Non-IT-related assets:**  
            - [threshold line]
        """

        self.NOVATION_RULES = """
            If the question concerns novation, amendment, or cancellation:
            - Use the exact policy title and approval structure from context.
            - Include deliberations, reviews, and co-management departments verbatim.
            """

        self.INSURANCE_DEPARTMENT_SUBTYPE_RULE = """
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
            """

        # ========== CLIENTS ==========
        self.client = CosmosClient(url=self.cosmos_endpoint, credential=self.cosmos_key)
        self.db = self._ensure_database(self.db_name)
        self.container = self._ensure_container(self.container_name)
        self.history_container = self._ensure_container(self.history_container_name, partition_key="user_id")

        # ========== EMBEDDINGS + LLM ==========
        self.embeddings = OpenAIEmbeddings(model=self.model_name)
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.1,
            api_key=self.OPENAI_API_KEY,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        # ========== RETRIEVER & RAG CHAIN ==========
        # (constructed dynamically when ask() is called)


    # ---------------------------
    # Small talk helper
    # ---------------------------
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
    # ---------------------------
    # CSV debug append
    # ---------------------------
    def _append_history_to_csv(self, user_id: str, memory_pairs: list):

        csv_file = "chat_history_log.csv"
        file_exists = os.path.exists(csv_file)
        now_ts = datetime.utcnow().isoformat()

        rows = []

        for qdata, assistant_answer in memory_pairs:

            # --- New dict format ---
            if isinstance(qdata, dict):
                original_q = qdata.get("original", "")
                rewritten_q = qdata.get("rewritten", "")   # <-- FIX: "question" holds rewritten
            else:
                # --- Old fallback format ---
                original_q = qdata
                rewritten_q = ""

            rows.append([
                now_ts,
                user_id,
                original_q,
                rewritten_q,
                assistant_answer
            ])

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "user_id",
                    "original_question",
                    "rewritten_question",
                    "assistant_answer"
                ])

            writer.writerows(rows)


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


    # ---------------------------
    # History (memory) handling
    # ---------------------------
    def load_user_history(self, user_id: str):
        """
        Load last N chat turns from Cosmos DB.
        Cosmos now stores each turn as ONE item:
            { user: "...", assistant: "..." }
        """
        try:
            # Fetch items in chronological order
            query = f"""
            SELECT TOP {self.history_limit} * FROM c
            WHERE c.user_id = '{user_id}'
            ORDER BY c.timestamp ASC
            """

            items = list(self.history_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))

            paired = []

            for it in items:
                user_q = it.get("user")
                assistant_a = it.get("assistant")

                # Only accept complete pairs
                if user_q and assistant_a:
                    paired.append((user_q, assistant_a))

            # Store to memory
            self.chat_memory[user_id] = deque(
                paired[-self.history_limit:],  # keep latest N
                maxlen=self.history_limit
            )

            print(f"✅ Loaded {len(self.chat_memory[user_id])} past turns for {user_id}")

        except Exception as e:
            print(f"❌ Error loading history for {user_id}: {e}")
            self.chat_memory[user_id] = deque(maxlen=self.history_limit)


    def save_to_memory(self, chat_state_memory, user_id: str, user_msg: str, assistant_msg: str):
        """Save chat turn in both session state and rolling memory; autosync every few turns."""


        # Build question record
        if hasattr(self, "_last_rewritten"):
            original_q = self._last_rewritten.get("original", user_msg)
            rewritten_q = self._last_rewritten.get("rewritten", None)
        else:
            original_q = user_msg
            rewritten_q = None

        # Store BOTH in a dict
        question_record = {
            "original": original_q,
            "rewritten": rewritten_q
        }
        # Initialize chat memory in both state and instance, if missing
        # Ensure memory exists
        if user_id not in self.chat_memory:
            self.chat_memory[user_id] = deque(maxlen=self.history_limit)
            self.last_sync_counter[user_id] = 0

        self.chat_memory[user_id][-1] = (question_record, assistant_msg)
        # Increment turn counter
        self.last_sync_counter[user_id] = self.last_sync_counter.get(user_id, 0) + 1

        # ✅ Auto-sync every N turns
        if self.last_sync_counter[user_id] >= self.autosave_interval:
            print(f"💾 Auto-syncing {user_id}'s chat history to Cosmos...")
            self.persist_user_history(user_id)
            self.last_sync_counter[user_id] = 0
        # Build updated memory dict for returning
        updated_memory = {**chat_state_memory}
        updated_memory[user_id] = list(self.chat_memory[user_id])
        return updated_memory


    def persist_user_history(self, user_id: str, state=None):
        """
        Persist last N turns to Cosmos DB.
        Works during autosave (RAG instance available) AND final exit (only state available).
        """
        try:
            # ✅ Determine memory source
            memory = []

            # 1️⃣ If instance memory exists and not empty
            if hasattr(self, "chat_memory") and self.chat_memory.get(user_id):
                memory = list(self.chat_memory[user_id])
                print(f"💾 Using in-memory chat cache for {user_id} ({len(memory)} turns).")
                print('IF LOOP')
            # 2️⃣ Otherwise fallback to state object
            elif state and hasattr(state, "state") and "chat_memory" in state.state:
                memory = state.state["chat_memory"].get(user_id, [])
                print(f"💾 Using state-based chat memory for {user_id} ({len(memory)} turns).")
                print('el LOOP')
            if not memory:
                print(f"⚠️ No chat history found to persist for {user_id}.")
                print('IF not memory LOOP')
                return

            total_turns = len(memory)
            last_saved = self.last_persist_index.get(user_id, 0)

            # -------- Determine new turns --------
            new_turns = memory[last_saved:]  # only unsaved items
            if not new_turns:
                print(f"ℹ️ No new turns to persist for {user_id}.")
                return

            print(f"💾 Persisting {len(new_turns)} NEW turns for {user_id}")
            # -------------------------
            # Append to local CSV debug log
            # -------------------------
            self._append_history_to_csv(user_id, new_turns)
            # -------------------------

            # ✅ Delete older records (keeping only last N turns)
            query = f"""
            SELECT c.id, c.timestamp FROM c 
            WHERE c.user_id = '{user_id}'
            ORDER BY c.timestamp DESC OFFSET {self.history_limit * 2} LIMIT 100
            """
            old_items = list(self.history_container.query_items(
                query=query, enable_cross_partition_query=True
            ))
            for it in old_items:
                try:
                    self.history_container.delete_item(it["id"], partition_key=user_id)
                except Exception:
                    pass  # tolerate partial deletion for safety
            if old_items:
                print(f"🧹 Pruned {len(old_items)} old messages for {user_id}.")

            # ✅ Save most recent N turns
            # Determine which question to save to DB

            for user_msg, assistant_msg in new_turns:
                if isinstance(user_msg, dict):
                    to_save = user_msg.get("rewritten") or user_msg.get("original")
                else:
                    to_save = user_msg
                item = {
                    "id": f"{user_id}-{datetime.utcnow().isoformat()}",
                    "user_id": user_id,
                    "user": to_save,
                    "assistant": assistant_msg,
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.history_container.upsert_item(item)

            self.last_persist_index[user_id] = total_turns
            print(f"✅ Persisted {len(new_turns)} messages for {user_id} to Cosmos.")

        except Exception as e:
            print(f"❌ Failed to persist history for {user_id}: {e}")


    def update_chat_memory(self, user_id: str, user_msg: str, assistant_msg: str):
        """Update in-memory history (no DB write during active chat)."""

        if user_id not in self.chat_memory:
            self.chat_memory[user_id] = deque(maxlen=self.history_limit)
        self.chat_memory[user_id].append((user_msg, assistant_msg))


    def clear_user_history(self, user_id: str):
        """
        Delete all chat history items for a given user_id from the Cosmos DB chat container.
        """
        try:
            # Query all items for this user_id
            query = f"SELECT c.id FROM c WHERE c.user_id = '{user_id}'"
            items = list(self.history_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            print(f"🧹 Found {len(items)} messages for user '{user_id}' to delete...")
            # exit()
            # Delete each item
            for item in items:
                item_id = item["id"]
                self.history_container.delete_item(item=item_id, partition_key=user_id)

            print(f"✅ Cleared all chat history for user '{user_id}'.")
            return True

        except Exception as e:
            print(f"❌ Error clearing history for {user_id}: {e}")
            return False


    def clean_content(self, doc: 'Document') -> str:
        """
        Safely extracts and cleans the text content from a LangChain Document object.

        Args:
            doc: A langchain_core.documents.base.Document object.

        Returns:
            The cleaned page_content string.
        """
        if not hasattr(doc, 'page_content') or not doc.page_content:
            return ""

        text = doc.page_content

        # 1. Replace symbols with standard list markers
        text = text.replace('◎', '*').replace('○', '*').replace('△', '*').replace(' ', '')

        # 2. REMOVE ALL CLASSIFICATION TAGS
        text = text.replace('(EXECUTIVEOFFICERS)', '').replace('(GLOBAL/REGIONALDIRECTORS)', '')

        # 3. Clean up phrasing (to isolate the roles more clearly)
        # text = text.replace(' is the Chairperson of the ShipManagementCommittee.', ' (Chairperson)')
        # text = text.replace('Members of the ShipManagementCommittee are:', 'Members:')
        # text = text.replace('Sub-members of the ShipManagementCommittee are:', 'Sub-members:')

        # 4. Strip extra whitespace that might have been introduced
        text = text.strip()

        return text


    def get_memory_pairs(self, user_id: str):
        """
        Load ALL memory pairs (ordered oldest → newest).
        Shared by rewriting + injection.
        Never formatted here.
        """
        memory_pairs = self.chat_memory.get(user_id, [])

        # If memory empty → fallback to Cosmos
        if not memory_pairs:
            items = list(self.history_container.query_items(
                query=f"SELECT * FROM c WHERE c.user_id='{user_id}' ORDER BY c.timestamp ASC",
                enable_cross_partition_query=True
            ))
            memory_pairs = [
                (item.get("user", ""), item.get("assistant", ""))
                for item in items if item.get("user") or item.get("assistant")
            ]
            self.chat_memory[user_id] = memory_pairs

        return memory_pairs


    def get_history_for_rewrite(self, user_id: str, turns: int = 1) -> str:
        """
        Returns the last N *completed* (user, assistant) turns.
        Used for pronoun rewriting and can be reused elsewhere safely.
        Ensures slicing works even if underlying memory is a deque.
        """

        pairs = self.chat_memory.get(user_id, [])

        # Convert deque → list for safe slicing
        if not isinstance(pairs, list):
            pairs = list(pairs)

        # Filter OUT incomplete turns: (question, None)
        completed_pairs = [(u, a) for u, a in pairs if a]

        if not completed_pairs:
            return ""

        # Get last N completed turns
        last_pairs = completed_pairs[-turns:]

        # Format as history text
        history_text = "\n".join(
            f"Human: {u}\nAI: {a}" for u, a in last_pairs
        )

        return history_text.strip()


    def get_history_for_injection(self, user_id: str, turns: int = 1) -> str:
        # Get in-memory history
        pairs = self.chat_memory.get(user_id, [])

        # Convert deque → list
        if not isinstance(pairs, list):
            pairs = list(pairs)

        # No history → return empty
        if not pairs:
            return ""

        # Extract last N turns safely
        last_pairs = pairs[-turns:]

        # Format
        history_text = "\n".join(
            f"Human: {u}\nAI: {a if a is not None else ''}"
            for u, a in last_pairs
        )

        return history_text.strip()


    # Step 3️⃣ — Inject history before prompt
    def inject_history(self, inputs: dict) -> dict:
        """
        Inject formatted chat history and context into the LLM input.
        Compatible with both LangGraph and standalone RAG pipeline.
        """

        # 🧩 Resolve user_id safely
        uid = inputs.get("user_id")
        rewritten_question = inputs["question"]   # now gets rewritten version 🎉

        if isinstance(uid, dict):
            uid = uid.get("user_id") or str(uid)
        if not isinstance(uid, str):
            uid = str(uid)

        # 🧠 Retrieve full chat history (hybrid: memory + Cosmos) for every question
        history = self.get_history_for_injection(uid, self.history_limit)

        # 📄 Handle Document objects or dict fallback
        context_docs = inputs.get("context", [])

        # iterates, cleans the content, and extracts it for formatting
        formatted_context = "\n\n".join([
            self.clean_content(d)  # <-- CALL USING self. AND PASS THE WHOLE OBJECT 'd'
            # The clean_content method handles accessing d.page_content internally
            for d in context_docs
            # Add a filter to only process valid LangChain Document objects
            if hasattr(d, 'page_content')
        ])

        return {
            "context": formatted_context,
            "question": rewritten_question,
            "history": history,
            "user_id": uid
        }


    def _add_retrieval_hints(self, text: str) -> str:
        """
        Adds structured retrieval hints (thresholds, IT vs Non-IT context, novation, golf, etc.)
        to bias vector search toward the correct policy section.
        """
        import re
        from datetime import datetime

        def _format_usd(n: int) -> str:
            s = f"{n:,}"
            return f"US$ {s}"

        normalized_hints = []
        pattern = (
            r"(?=.*\bservice\s+agreement\b)"
            r"(?=.*\b(conclusion|terminate|termination|revise|revision|concluding|revising|terminating)\b)"
            r"(?=.*\b(approval|approve|authorization|authorize|authorisation|approver|department|criteria)\b)"
        )
        if re.search(pattern, text.lower(), flags=re.I):
            print("✅ Detected 'Conclusion / Termination / Revision of Service Agreement' session.")
            normalized_hints = ['Conclusion/Termination/Revision of service agreement with MCTSPR subsidiaries']
            return "\n\nHINTS: " + "; ".join(list(set(normalized_hints)))

        amount_value = None

        # --- Extract amount ---
        m = re.search(r"(?:US\$|USD|\$)\s*([0-9]{1,3}(?:[, ]?[0-9]{3})*|[0-9]+)", text, flags=re.I)
        if m:
            raw = m.group(1)
            amount_value = int(re.sub(r"[^0-9]", "", raw)) if raw else None
            if amount_value is not None:
                normalized_hints.append(_format_usd(amount_value))

                # Threshold categories
                if amount_value < 25000:
                    normalized_hints.append("Less than US$25,000")
                elif amount_value < 50000:
                    normalized_hints.append("Less than US$50,000")
                else:
                    if re.search(r"less than|under", text, flags=re.I) and amount_value == 50000:
                        normalized_hints.append("Less than US$50,000")
                    else:
                        normalized_hints.append("US$50,000 or more")
        # --- Golf membership context ---
        if re.search(r"golf\s*(course)?\s*membership", text, re.I):
            normalized_hints += [
                "Golf course membership",
                "Non-IT fixed assets",
                "Acquisition or disposal of assets excluding IT",
                "Equipment and fixtures",
            ]

        # --- Write-off / FCCSP migrated ---
        if re.search(r"(write[- ]?off|migrated|fccsp)", text, re.I):
            if amount_value and amount_value < 25000:
                normalized_hints += [
                    "Write-off of assets",
                    "Follow same approval criteria by amount",
                    "Less than US$25,000 — A4 approval within department",
                    "No formal GPM submission required",
                ]
            else:
                normalized_hints += [
                    "Write-off of assets",
                    "Approval required as per disposal threshold",
                    "Check golf course membership disposal rule",
                ]

        if re.search(r"\bcommittee\b", text, re.I):
            normalized_hints += [
                "Committee structure",
                "Committee composition",
                "Chairperson",
                "Members",
                "Sub-members",
                "Secretariat",
                "Department roles",
            ]
        # --- Deduplication ---
        if not normalized_hints:
            normalized_hints = extract_significant_words(text)
            if 'insurance' in text.lower():
                normalized_hints += ['responsible dept for insurance']
        if normalized_hints:
            hints = []
            seen = set()
            for h in normalized_hints:
                if h not in seen:
                    hints.append(h)
                    seen.add(h)
            text += "\n\nHINTS: " + "; ".join(hints)

        return text


    # ------------------------------------------------------------
    # VECTOR SEARCH ON COSMOS
    # ------------------------------------------------------------
    def _embed_query(self, query: str):
        """Get OpenAI embeddings for the query."""
        return self.embeddings.embed_query(query)

    def _merge_and_dedup_chunks(self, main_text: str, neighbor_text: str):
        """Merge only main + its neighbor; dedupe overlapping."""
        if not neighbor_text:
            return main_text

        main_lines = [l.rstrip() for l in main_text.splitlines()]
        neigh_lines = [l.rstrip() for l in neighbor_text.splitlines()]

        # Normalize for dedupe
        def norm(x):
            x = x.strip().lower()
            x = re.sub(r"[\*\-\u2022\u25AA\u25CF•●]", "", x)
            x = re.sub(r"\s+", " ", x)
            x = x.replace("us$", "$").replace("jp¥", "¥")
            return x

        main_norm = [norm(l) for l in main_lines]
        neigh_norm = [norm(l) for l in neigh_lines]

        # Remove tail->head overlap (simple boundary match)
        max_overlap = min(20, len(main_norm), len(neigh_norm))

        trimmed_main = main_lines
        for k in range(max_overlap, 0, -1):
            if main_norm[-k:] == neigh_norm[:k]:
                trimmed_main = main_lines[:-k]
                break

        # Remove neighbor lines that already appear in main
        seen = set(norm(l) for l in trimmed_main)
        cleaned_neighbor = []
        for raw, nr in zip(neigh_lines, neigh_norm):
            if nr not in seen:
                cleaned_neighbor.append(raw)
                seen.add(nr)

        return "\n".join(trimmed_main + cleaned_neighbor)


    def _fetch_chunk_by_seq(self, doc_id: str, seq: int):
        """Fetch a specific chunk by doc_id + seq."""
        sql = """
        SELECT TOP 1 c.id, c.text, c.metadata, c.source_doc_id, c.chunk_index
        FROM c 
        WHERE c.source_doc_id = @doc_id AND c.chunk_index = @seq
        """
        params = [
            {"name": "@doc_id", "value": doc_id},
            {"name": "@seq", "value": seq},
        ]

        items = list(self.container.query_items(
            query=sql,
            parameters=params,
            enable_cross_partition_query=True
        ))

        return items[0] if items else None



    def search_cosmos_documents(self, query: str):
        """Perform vector search using Cosmos SQL API's VectorDistance function."""
        query_emb = self._embed_query(query)
        emb_json = json.dumps(query_emb)

        sql = f"""
        SELECT TOP {self.top_k}
            c.id, c.text, c.source_doc_id, c.chunk_index,
            VectorDistance(c.vector_embedding, {emb_json}) AS score
        FROM c
        ORDER BY VectorDistance(c.vector_embedding, {emb_json})
        """

        items = list(self.container.query_items(
            query=sql,
            enable_cross_partition_query=True
        ))

        # Group by source_doc_id
        grouped = {}
        for item in items:
            doc_id = item["source_doc_id"]
            grouped.setdefault(doc_id, []).append(item)

        final_documents = []

        for doc_id, chunks in grouped.items():
            merged_blocks = []

            # Sort by sequence number
            chunks_sorted = sorted(chunks, key=lambda x: x["chunk_index"])

            for item in chunks_sorted:
                seq = item["chunk_index"]
                main_text = item["text"]

                # Fetch NEXT chunk
                next_item = self._fetch_chunk_by_seq(doc_id, seq + 1)
                next_text = next_item.get("text", "") if next_item else ""

                # Merge + dedup
                merged_text = self._merge_and_dedup_chunks(main_text, next_text)

                merged_blocks.append((seq, merged_text))

            # Final merge of multiple selected blocks in same doc
            merged_blocks.sort(key=lambda x: x[0])

            final_text = ""
            for _, block_text in merged_blocks:
                if not final_text:
                    final_text = block_text
                else:
                    final_text = self._merge_and_dedup_chunks(final_text, block_text)

            final_documents.append(
                Document(
                    page_content=final_text,
                    metadata={"doc_id": doc_id}
                )
            )

        return final_documents


    # ------------------------------------------------------------
    # PROMPT + RAG CHAIN
    # ------------------------------------------------------------
    @staticmethod
    def format_docs(docs):
        """Combine retrieved docs into one context block."""
        with open("debug_docs.txt", "w") as f:
            f.write("\n\n".join([doc.page_content for doc in docs]))
        return "\n\n".join([doc["page_content"] for doc in docs])

    def _build_prompt_template(self, question: str) -> "ChatPromptTemplate":
        """
        Create a prompt that includes chat history + context + dynamic extra rules.
        Automatically detects session (Service Agreement, Insurance, Charter, etc.)
        and applies the appropriate rule sets while avoiding conflicts like cost overlap.
        """

        # ==============================
        # CORE RULE DEFINITIONS
        # ==============================

        self.HARD_EXCLUSION_RULE = """
        CRITICAL PRE-FILTER:
        If the question contains 'Less than' or 'Under', you are FORBIDDEN from selecting
        any policy line that contains 'or more'. Ignore those lines regardless of the amount.
        """

        self.FINAL_PRIORITIZATION_RULE = """
        CRITICAL FINANCIAL PRIORITY RULE:
        When multiple monetary thresholds appear (e.g., US$25,000, US$50,000, US$500,000):
        1. Always apply numeric reasoning based on the question’s hinted amount.
        2. Ignore any line containing 'or more' when the question includes 'Less than' or smaller amounts.
        3. Select the lowest valid threshold that covers the amount (most restrictive).
        4. Never choose higher thresholds because they look more detailed.
        5. Match exactly the numeric range implied by the question.
        """

        self.RANGE_SELECTION_RULE = """
        DURATION RANGE SELECTION RULE:
        If the question specifies a duration (e.g., 'for 3 years', '2 years period', '48 months'):
        Identify which policy line covers that duration and select only that line.
        """

        self.DURATION_RANGE_CONTAINMENT_RULE = """
        DURATION RANGE CONTAINMENT:
        If the question specifies a duration (years/months/period),
        select only the policy line whose numeric range *contains* that duration.
        """

        self.NUMERIC_SPECIFICITY_TIEBREAK = """
        NUMERIC SPECIFICITY TIE-BREAK:
        When multiple lines include the value, prefer the narrower (more specific) range.
        """

        self.RELEVANT_LINE_FILTER_RULE = """
        CRITICAL OUTPUT CLEANUP RULE:
        Include only the minimum lines directly answering the question.
        Omit category headers unless needed to clarify IT vs Non-IT context.
        """

        self.SERVICE_AGREEMENT_RULE = """
        ==============================
        ### SERVICE AGREEMENT SESSION LOGIC
        ==============================
        If the question concerns **Conclusion**, **Termination**, or **Revision** of a Service Agreement:
        1. Identify all departments and approvers responsible.
        2. Use only the policy section referring to **service agreement (conclusion/termination/revision) with MCTSPR subsidiaries**.
        3. Do not mix with other contract types.
        4. Include deliberation, review, and co-management exactly as stated.
        """

        self.CONTEXTUAL_FILTERING_RULES = """
        ==============================
        ### CONTEXTUAL FILTERING RULES
        ==============================
        10. If the question explicitly mentions a **policy year** (e.g., “Policy Year 2025”):
            - Interpret it as referring to the **annual plan** for that policy year.  
            - Only include approval criteria applicable to the annual plan.  
            - **Exclude unrelated categories** like “Important” or “Others” unless explicitly required by the question.

        11. If a **policy year** is not specified, apply general approval criteria relevant to the subject matter.
        """
        FORMATTING_RULES = """
                ==============================
                ### OUTPUT FORMAT RULES
                ==============================
                You MUST format the final answer EXACTLY as follows:

                1. Avoid unnecessary explanations.
                    - DO NOT explain rules again.
                    - DO NOT restate numeric logic.
                    - ONLY present the extracted policy content.
                2. Use bullet points (“- ”) for listing approvers, departments, members, or thresholds.

                3. 
                    a) if multiple headers involves 
                        Use clear section headers (Only include sections that are relevant.):
                        ** Fixed Assets (Excluding IT-related assets)**
                        ** IT-related Fixed Assets**
                        ** Service Agreement**
                        
                    b) else no change required here.

                4. Add a short final summary only when multiple thresholds are listed.
                    Format it like this:
                    **Summary:**
                    <text>

                """
        # ==============================
        # ADDITION: Minimal Multi-Topic Cost Awareness
        # ==============================
        def _detect_cost_topics(q: str):
            q = q.lower()
            topics = []
            if re.search(r"(implement|system|software|development|upgrade|installation|new\s+system)", q):
                topics.append("implementation")
            if re.search(r"(maintenance|support|service contract|renewal|annual)", q):
                topics.append("maintenance")
            if re.search(r"(acquisition|purchase|procurement|new product)", q):
                topics.append("acquisition")
            if re.search(r"(disposal|write[- ]off|sell)", q):
                topics.append("disposal")
            return topics

        # Detect cost topics for later hinting
        detected_topics = _detect_cost_topics(question)

        # ==============================
        # SESSION PROFILES
        # ==============================
        self.SESSION_PROFILES = {
            "service_agreement": {
                "pattern": r"(?=.*\bservice\s+agreement\b)(?=.*\b(conclusion|terminate|termination|revise|revision|concluding|revising|terminating)\b)",
                "rules_add": ["SERVICE_AGREEMENT_RULE"],
                "rules_block": ["COST_RULES"],
                "priority": 3,
                "description": "Conclusion / Termination / Revision of Service Agreements"
            },
            "insurance": {
                "pattern": r"\b(insurance|policy year|premium|renewal|p&i|cover|cli|fdd|tcl|dth)\b",
                "rules_add": ["INSURANCE_DEPARTMENT_SUBTYPE_RULE", "CONTEXTUAL_FILTERING_RULES"],
                "rules_block": [],
                "priority": 2,
                "description": "Insurance and Subtype Approvals"
            },
            "charter_duration": {
                "pattern": r"\b(charter in|charter out|bare boat|time charter|period|year|month)\b",
                "rules_add": ["RANGE_SELECTION_RULE", "DURATION_RANGE_CONTAINMENT_RULE", "NUMERIC_SPECIFICITY_TIEBREAK"],
                "rules_block": [],
                "priority": 1,
                "description": "Charter contract duration-based approvals"
            },
        }

        
        # ==============================
        # SESSION DETECTION LOGIC
        # ==============================
        q_lower = question.lower()
        matched_sessions = []
        for name, profile in self.SESSION_PROFILES.items():
            if re.search(profile["pattern"], q_lower, flags=re.I):
                matched_sessions.append((profile["priority"], name, profile))

        session_name, session_profile = (None, None)
        if matched_sessions:
            matched_sessions.sort(reverse=True)
            _, session_name, session_profile = matched_sessions[0]
            print(f"✅ Detected session: {session_profile['description']}")
        else:
            print("ℹ️ No specific session detected; applying generic logic.")

        # ==============================
        # DYNAMIC RULE BUILDING
        # ==============================
        extra_rules = ""
        blocked = set(session_profile["rules_block"]) if session_profile else set()

        # Add session-specific rules first (if any)
        if session_profile:
            for rule in session_profile["rules_add"]:
                extra_rules += getattr(self, rule)

        # Apply generic rules only if not blocked by session
        if "COST_RULES" not in blocked and any(x in q_lower for x in ["cost", "fee", "amount", "budget", "it-related", "acquisition", "disposal"]):
            print("Applying cost rules\n")
            # extra_rules += self.COST_RULES

        # Duration-based logic
        if re.search(r"\b(year|years|month|months|period)\b", q_lower, flags=re.I):
            print("Applying RANGE_SELECTION_RULE ,DURATION_RANGE_CONTAINMENT_RULE, NUMERIC_SPECIFICITY_TIEBREAK\n")
            extra_rules += self.RANGE_SELECTION_RULE
            extra_rules += self.DURATION_RANGE_CONTAINMENT_RULE
            extra_rules += self.NUMERIC_SPECIFICITY_TIEBREAK

        # Add contextual filtering when "policy year" appears (global override)
        if re.search(r"\bpolicy\s+year\b", q_lower, flags=re.I):
            extra_rules += self.CONTEXTUAL_FILTERING_RULES

        # Insurance subtype logic (only if not handled by service_agreement)
        if not session_profile or session_name != "service_agreement":
            if re.search(r"\b(insurance|policy year|premium|renewal|p&i|cover)\b", q_lower, flags=re.I):
                extra_rules += self.INSURANCE_DEPARTMENT_SUBTYPE_RULE
            elif re.search(r"\b(cli|fdd|tcl|dth)\b", q_lower, flags=re.I):
                extra_rules += self.INSURANCE_DEPARTMENT_SUBTYPE_RULE
                
        

        # Always add final cleanup rule
        # extra_rules += self.RELEVANT_LINE_FILTER_RULE

        # ==============================
        # FINAL PROMPT TEMPLATE
        # ==============================
        template = f"""
        You are **Thinkpalm’s Corporate Knowledge Assistant**.
        Your job is to produce an **exact, policy-faithful answer** using *only* the information from the Document Context below.

        Guidelines:
        1. Use the conversation history below to understand the topic and follow-up questions.
        2. If the question refers to "it", "this", or "explain again", look at the last assistant response in the history.
        3. Only use the provided context for factual information — do not invent details.
        4. Answer **only** from the provided context.
            - If not enough info exists, reply exactly:  
            "I do not have sufficient information in the available policy context to answer that."
        5. **CRITICAL - Amount Threshold Selection and Range Interpretation**:
            a) When a SPECIFIC amount is given (e.g., $8,300):
                - Identify ALL thresholds that this amount qualifies for
                - Select the MOST SPECIFIC threshold that applies
                - Example: $8,300 qualifies for both "Less than US$50,000" and "Less than US$25,000"
                - Use "Less than US$25,000" (more specific), NOT "Less than US$50,000"

            b) When a RANGE is given (e.g., "less than US$50,000"):
                - **CRITICAL**: Show ONLY thresholds that fall WITHIN that range
                - "Less than US$50,000" means amounts from $0 to $49,999
                - DO NOT include "US$50,000 or more" threshold - that is OUTSIDE the range
                - DO include: "US$25,000 or more", "US$10,000 or more", "Less than US$10,000" (all are within the range)
                - For IT-related assets with "less than $50,000": show US$25,000 or more, US$10,000 or more, and Less than US$10,000
                - For non-IT assets with "less than $50,000": show only "Less than US$50,000" (if $25,000+ show "Less than US$25,000")

            c) Boundary Rules:
                - "Less than X" = amounts below X (does NOT include X)
                - "X or more" = amounts at or above X (INCLUDES X)
                - Never include a threshold that requires amounts AT or ABOVE the upper limit of a range
        6. **CRITICAL - Term Mapping for Implementation and Maintenance**:
            a) **"Implementation"** queries should be treated as **Acquisition of Fixed Assets**:
                - If IT context (software/system): Map to "Acquisition of IT related fixed assets"
                - If non-IT context or unclear: Present both IT and non-IT acquisition rules
                - Example: "implementation cost" = acquisition cost

            b) **"Maintenance"** queries should be treated as **Service Agreements**:
                - If IT context (software/system): Map to "IT-related service agreements"
                - If non-IT context or unclear: Present both IT and non-IT service agreement rules
                - Example: "maintenance cost" = service agreement/contract cost

            c) **Both Implementation AND Maintenance** in same query:
                - Present BOTH sections: Acquisition rules AND Service agreement rules
                - Clearly separate: "(A) Implementation (Acquisition): ..." and "(B) Maintenance (Service Agreement): ..."
                - Example: "implementation $40K and maintenance $30K" → show acquisition rules for $40K AND service agreement rules for $30K
        7. **CRITICAL - Asset Category Coverage**: When the question asks about generic "acquisition", "product", or "purchase" WITHOUT explicitly specifying IT or non-IT:
            - You MUST check if BOTH "Acquisition of assets (Excluding IT related assets)" AND "Acquisition of IT related assets" are present in the Document Context
            - If BOTH categories exist, you MUST present BOTH categories in your answer with clear labeling
            - Format: Present as "(A) Fixed Assets (excluding IT-related assets): [approval details]" and "(B) IT-related Fixed Assets: [approval details]"
            - Do NOT assume the question is only about one category unless explicitly stated
            - Example for "less than US$50,000":
                * (A) Fixed Assets (excluding IT-related assets): If exact threshold exists, show it → "Less than US$50,000: A4; Review: GPM; CC: GAF"
                * (B) IT-related Fixed Assets: For amounts less than US$50,000, show applicable sub-thresholds within that range:
                - US$25,000 or more (but < US$50,000): A3; Co-Management Dept.: ICS / DXS; Review: GPM; CC: GAF
                - US$10,000 or more (but < US$25,000): A3; Co-Management Dept.: ICS / DXS; CC: GAF
                - Less than US$10,000: A4; Co-Management Dept.: ICS / DXS; CC: GAF

        8. After all range interpretation and reasoning , show the most specific threshold and minimal explanation / summary at the end , making sure of non-repeated information.
        {extra_rules}
        {FORMATTING_RULES}
        

        ==============================
        ### QUESTION
        ==============================
        {question}

        ==============================
        ### DOCUMENT CONTEXT
        ==============================
        {{context}}

        ==============================
        ### MULTIPLE CATEGORY HANDLING
        ==============================
        If multiple applicable policy sections (e.g., IT-related and Excluding IT-related) are found,
        list each separately using clear headers ("For IT-related assets", "For Non-IT-related assets").
        Do not merge their details.

        ==============================
        ### OUTPUT
        ==============================
        Answer:
        """

        return ChatPromptTemplate.from_template(template.replace("{extra_rules}", extra_rules))



    def _build_rag_chain(self, question):
        """Create the retrieval + generation chain with selective memory-based rewriting."""

        PRONOUN_PATTERN = re.compile(
            r"\b("
            r"it|its|it's|it’s|"         # it → possessive + contractions
            r"this|that|these|those|"    # demonstratives
            r"they|them|their|theirs|they're|theyre|"  # plurals + possessive
            r"he|his|him|"               # male
            r"she|her|hers|"             # female
            r"there|here|"               # vague locatives
            r"mentioned|above|same"      # generic vague referents
            r")\b",
            re.IGNORECASE
        )

        def rewrite_question(inputs):
            user_id = inputs["user_id"]
            question = inputs["question"].strip()
            history_text = self.get_history_for_rewrite(user_id, turns=1)

            # 🧩 If no prior history, skip rewriting
            if not history_text:
                self._last_rewritten = {
                    "rewritten": None,
                    "original": question,
                    "history_used": None
                }
                return {"user_id": user_id, "question": question}

            # 🧠 If pronoun detected → use LLM to clarify
            if PRONOUN_PATTERN.search(question):
                reformulation_prompt = f"""
                You are a helpful assistant. Rewrite the latest user question so it becomes 
                self-contained and unambiguous, based on the conversation history.

                REWRITE RULES:
                - Remove ALL pronouns (it, this, that, they, them, etc.).
                - EXCEPTION: The acronym “IT” referring to Information Technology **must be uppercase**.
                - Never output “it” in lowercase unless it is part of another word 
                (e.g., "acquisition", "notification").
                - If the context refers to “IT assets”, rewrite it EXACTLY as “IT assets”.
                - NEVER output the phrase “it asset acquisition” — convert it to “IT asset acquisition”.

                Conversation History:
                {history_text}

                Latest User Question: {question}

                Rewritten question:

                """
                rewritten = self.llm.invoke(reformulation_prompt).content.strip()
                print(f"🔁 Rewritten question: {rewritten}")
                self._last_rewritten = {
                    "rewritten": rewritten,
                    "original": question,
                    "history_used": history_text
                }
                return {"user_id": user_id, "question": rewritten}

            # 🚀 No pronouns → no rewrite
            self._last_rewritten = {
                "rewritten": None,
                "original": question,
                "history_used": history_text
            }
            return {"user_id": user_id, "question": question}


        # Step 2️⃣ — Retrieval runnable (uses rewritten question)
        def retrieve_with_rewrite(inputs):
            """Rewrite the question, then search Cosmos with rewritten text."""
            rewritten = rewrite_question(inputs)

            # 🧠 Enrich the question with retrieval hints
            hinted_query = self._add_retrieval_hints(rewritten["question"])
            # print(f"[DEBUG] Retrieval query with hints:\n{hinted_query}")

            # 🔍 Use enriched question text for vector search
            # docs = self.search_cosmos_documents(rewritten["question"])
            docs = self.search_cosmos_documents(hinted_query) 

            for i, d in enumerate(docs):
                preview = d.page_content.replace("\n", " ").strip()[:180]
                score = d.metadata.get("score", 0)
                print(f"[DEBUG] Doc {i+1} | Score: {score:.3f} | Snippet: {preview}...")

            # Save full retrieval trace for external inspection
            with open("Verification_retrieved_docs.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n==============================\n")
                f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Rewritten Query: {rewritten['question']}\n")
                for i, d in enumerate(docs):
                    text = d.page_content
                    score = d.metadata.get("score", 0)
                    f.write(f"Doc {i+1} (Score {score:.3f}):\n{text}\n\n")
                f.write(f"Hinted query: {hinted_query}\n")

            return {
                "context": docs,
                "user_id": rewritten["user_id"],
                "question": rewritten["question"],
            }

        prompt = self._build_prompt_template(question)

        # Step 4️⃣ — Full chain
        return (
            RunnableLambda(retrieve_with_rewrite)
            | RunnableLambda(self.inject_history)
            | prompt
            | self.llm
            | StrOutputParser()
        )


    # ------------------------------------------------------------
    # ASK (MAIN ENTRYPOINT)
    # ------------------------------------------------------------
    def ask(self, user_id, question: str):
        """Run full RAG flow: retrieve from Cosmos + generate answer."""
        # -----------------------------
        # 0️⃣ SMALL TALK CHECK (NEW)
        # -----------------------------
        is_small_talk = self._is_small_talk(question)
        print("is_small_talk:", is_small_talk)
        
        if is_small_talk:
            # --- Small Talk Handling ---
            reply = self._generate_small_talk_response(question)

            # reply = "Hello! How can I help you today?"

            # Prepare rewritten/original record
            q_record = {
                "original": question,
                "rewritten": None,
                "history_used": ""
            }

            # --- SAFE MEMORY UPDATE (no placeholder logic needed) ---
            if user_id not in self.chat_memory:
                self.chat_memory[user_id] = deque(maxlen=self.history_limit)
            
            self.chat_memory[user_id].append((q_record, reply))

            # update rewrite tracker
            self._last_rewritten = {
                "original": question,
                "rewritten": None,
                "history_used": ""
            }

            return reply, [], q_record
        # 1️⃣ Load memory FIRST (only if empty)
        # 1️⃣ Load memory FIRST (only if empty)

        if user_id not in self.chat_memory or not self.chat_memory[user_id]:
            self.load_user_history(user_id)

        # 2️⃣ Insert placeholder BEFORE rewrite + retrieval
        self.chat_memory[user_id].append((question, None))
        print(f"💬 Asking: {question}")

        inputs = {"user_id": user_id, "question": question}
        self.rag_chain = self._build_rag_chain(question)
        response = self.rag_chain.invoke(inputs)

        docs = self.search_cosmos_documents(question)
        with open("Verification_retrieved_docs.txt", "a", encoding="utf-8") as f:
            f.write(f"Assistant: {response}\n")
            f.write(f"\n\n==============================\n")

        rewritten_q = getattr(self, "_last_rewritten", {}).get("rewritten", question) or question

        return response, docs, rewritten_q


