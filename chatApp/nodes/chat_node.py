from nodes.new_base_node import BaseNode
from utils.error_handling import PipelineState
from chains_old.retrievel_chain import ThinkpalmRAG

# path to your ThinkpalmRAG class
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaEmbeddings
import numpy as np
from langchain_openai import ChatOpenAI
# from langchain.evaluation.qa import QAEvalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
import os
import re
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

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
from langchain_core.prompts import ChatPromptTemplate
from collections import deque
from azure.cosmos import CosmosClient, exceptions, PartitionKey
import json
import uuid
load_dotenv()

import spacy
nlp = spacy.load("en_core_web_sm")

def extract_significant_words(sentence: str):
    doc = nlp(sentence)
    return [token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "NUM"} and not token.is_stop]
         
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
        self.autosave_interval = 10
        
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

    def load_user_history(self, user_id: str):
        """Load last 5 chat turns for a user from Cosmos DB into memory."""
        try:
            # load_user_history()
            query = f"""
            SELECT TOP {self.history_limit * 2} c.user, c.assistant, c.timestamp
            FROM c
            WHERE c.user_id = '{user_id}'
            ORDER BY c.timestamp ASC
            """
            items = list(self.history_container.query_items(query=query, enable_cross_partition_query=True))

            self.chat_memory[user_id] = deque(
                [(it["user"], it["assistant"]) for it in items[-self.history_limit:]],
                maxlen=self.history_limit
            )

            items = list(self.history_container.query_items(query=query, enable_cross_partition_query=True))

            self.chat_memory[user_id] = deque(
                [(it["user"], it["assistant"]) for it in items[-self.history_limit:]],
                maxlen=self.history_limit
            )
            print(f"✅ Loaded {len(self.chat_memory[user_id])} past turns from Cosmos for {user_id}")
        except exceptions.CosmosResourceNotFoundError:
            print(f"⚠️ No existing history for {user_id}.")
            self.chat_memory[user_id] = deque(maxlen=self.history_limit)
        except Exception as e:
            print(f"Error loading history for {user_id}: {e}")
            self.chat_memory[user_id] = deque(maxlen=self.history_limit)

    def save_to_memory(self, state, user_id: str, user_msg: str, assistant_msg: str):
        """Save chat turn in both session state and rolling memory; autosync every few turns."""
        
        # Initialize chat memory in both state and instance, if missing
        if "chat_memory" not in state:
            state["chat_memory"] = {}

        if user_id not in self.chat_memory:
            self.chat_memory[user_id] = deque(maxlen=self.history_limit)
            self.last_sync_counter[user_id] = 0

        # Add message to deque (auto-trims old entries)
        self.chat_memory[user_id].append((user_msg, assistant_msg))

        # Reflect it back to state (optional, for graph continuity)
        state["chat_memory"][user_id] = list(self.chat_memory[user_id])

        # Increment turn counter
        self.last_sync_counter[user_id] = self.last_sync_counter.get(user_id, 0) + 1

        # ✅ Auto-sync every N turns
        if self.last_sync_counter[user_id] >= self.autosave_interval:
            print(f"💾 Auto-syncing {user_id}'s chat history to Cosmos...")
            self.persist_user_history(user_id)
            self.last_sync_counter[user_id] = 0


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

            # 2️⃣ Otherwise fallback to state object
            elif state and hasattr(state, "state") and "chat_memory" in state.state:
                memory = state.state["chat_memory"].get(user_id, [])
                print(f"💾 Using state-based chat memory for {user_id} ({len(memory)} turns).")

            if not memory:
                print(f"⚠️ No chat history found to persist for {user_id}.")
                return

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
            for user_msg, assistant_msg in list(memory)[-self.history_limit:]:
                item = {
                    "id": f"{user_id}-{datetime.utcnow().isoformat()}",
                    "user_id": user_id,
                    "user": user_msg,
                    "assistant": assistant_msg,
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.history_container.upsert_item(item)

            print(f"✅ Persisted {len(memory)} messages for {user_id} to Cosmos.")
        
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

    # Step 3️⃣ — Inject history before prompt
    def inject_history(self, inputs: dict) -> dict:
        """
        Inject formatted chat history and context into the LLM input.
        Compatible with both LangGraph and standalone RAG pipeline.
        """

        # 🧩 Resolve user_id safely
        uid = inputs.get("user_id")
        if isinstance(uid, dict):
            uid = uid.get("user_id") or str(uid)
        if not isinstance(uid, str):
            uid = str(uid)

        # 🧠 Retrieve full chat history (hybrid: memory + Cosmos) for every question
        history = self.get_chat_history_text(uid)

        # 📄 Handle Document objects or dict fallback
        context_docs = inputs.get("context", [])
        
        # --- ✂️ CLEANUP PERFORMED HERE DURING STRING JOINING ---
        
        
        # This line iterates, cleans the content, and extracts it for formatting

    
        formatted_context = "\n\n".join([
            self.clean_content(d)  # <-- CALL USING self. AND PASS THE WHOLE OBJECT 'd'
            # The clean_content method handles accessing d.page_content internally
            for d in context_docs 
            # Add a filter to only process valid LangChain Document objects
            if hasattr(d, 'page_content')
        ])
        # print(context_docs)
        # print(formatted_context)

        return {
            "context": formatted_context,
            "question": inputs.get("question", ""),
            "history": history,
            "user_id": uid
        }
    def get_chat_history_text(self, user_id: str) -> str:
        """
        Retrieve chat history for the given user.
        Prefer in-memory (session) chat_memory; fallback to Cosmos DB.
        Returns formatted conversation string.
        """
        try:
            # ✅ Prefer in-memory cache
            memory = self.chat_memory.get(user_id, [])
            
            # 🧩 Defensive fix: ensure it's a list
            if isinstance(memory, dict):
                # Convert dict to list of (user, assistant) pairs if needed
                memory = [(k, v) for k, v in memory.items()]
            elif not isinstance(memory, list):
                memory = []

            # ⚙️ If empty → fallback to Cosmos DB
            if not memory:
                query = f"""
                SELECT TOP {self.history_limit * 2} * FROM c 
                WHERE c.user_id = '{user_id}'
                ORDER BY c.timestamp ASC
                """
                items = list(self.history_container.query_items(
                    query=query,
                    enable_cross_partition_query=True
                ))
                items = sorted(items, key=lambda x: x.get("timestamp", ""), reverse=False)
                print(f"✅ Loaded {len(items)} past turns from Cosmos for {user_id}")
                
                # ✅ Store into in-memory cache
                memory = [(i.get("user", ""), i.get("assistant", "")) for i in items]
                self.chat_memory[user_id] = memory
            else:
                print(f"💾 Loaded {len(memory)} turns from in-memory cache for {user_id}")

            # ✅ Safely slice only lists
            limited_pairs = memory[-self.history_limit:] if isinstance(memory, list) else []
            
            history_text = "\n".join([
                f"User: {u}\nAssistant: {a}" for u, a in limited_pairs
            ])
            return history_text.strip()

        except Exception as e:
            print(f"⚠️ Error retrieving chat history for {user_id}: {e}")
            return ""

    def _add_retrieval_hints(self, text: str) -> str:
        """
        Adds structured retrieval hints (thresholds, IT vs Non-IT context, novation, golf, etc.)
        to bias vector search toward the correct policy section.
        """
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
                # Threshold categories
                if amount_value < 25000:
                    normalized_hints.append("Less than US$25,000")
                elif amount_value < 50000:
                    normalized_hints.append("Less than US$50,000")
                else:
                    # Check if the query specifically mentioned "less than" this amount
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
                normalized_hints+=['responsible dept for insurance']
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
        

        # ⚡️ Pronoun / vague reference detector
        PRONOUN_PATTERN = re.compile(
            r"\b(it|this|that|they|them|those|there|these|he|she|his|her|their|mentioned|above|same)\b",
            re.IGNORECASE
        )

        def rewrite_question(inputs):
            user_id = inputs["user_id"]
            question = inputs["question"].strip()
            history_text = self.get_chat_history_text(user_id)

            # 🧩 If no prior history, skip rewriting
            if not history_text:
                return {"user_id": user_id, "question": question}

            # 🧠 If pronoun detected → use LLM to clarify
            if PRONOUN_PATTERN.search(question):
                reformulation_prompt = f"""
                You are a helpful assistant. Based on the conversation below,
                rewrite the latest user question so that it is self-contained and unambiguous.

                Conversation History:
                {history_text}

                Latest User Question: {question}

                Rewritten question:
                """
                rewritten = self.llm.invoke(reformulation_prompt).content.strip()
                print(f"🔁 Rewritten question: {rewritten}")
                return {"user_id": user_id, "question": rewritten}

            # 🚀 No pronouns → no rewrite
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
                # exit()
                    
                
            return {
                "context": docs,
                "user_id": rewritten["user_id"],
                "question": rewritten["question"],
            }

        prompt = self._build_prompt_template(question)
        
                

        
        # Step 4️⃣ — Full chain
        return (
            RunnableLambda(retrieve_with_rewrite)
            |  RunnableLambda(lambda inputs: self.inject_history(inputs))
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------
    # ASK (MAIN ENTRYPOINT)
    # ------------------------------------------------------------
    def ask(self, user_id, question: str):
        """Run full RAG flow: retrieve from Cosmos + generate answer."""
        if user_id not in self.chat_memory:
            # Load at the initial time. per user per session
            self.load_user_history(user_id)
            
        print(f"💬 Asking: {question}")
        
        inputs = {"user_id": user_id, "question": question}
        self.rag_chain = self._build_rag_chain(question)
        response = self.rag_chain.invoke(inputs)
        
        docs = self.search_cosmos_documents(question)
        with open("Verification_retrieved_docs.txt", "a", encoding="utf-8") as f:
            f.write(f"Assistant: {response}\n")
            f.write(f"\n\n==============================\n")

        
        # self.update_chat_memory(user_id, question, response)
        
        return response, docs
    


class RetrieverNode(BaseNode):
    def __init__(self):
        super().__init__("RetrieverNode")
        self.rag_bot = ThinkpalmRAG()
        
    def execute(self, state: "ChatState") -> "ChatState":
        # --- Run RAG ---
        initial_answer, retrieved_docs = self.rag_bot.ask(
            state.user_id,
            state.question
        )

        # --- Save memory (uses dict-like input, so convert state to dict) ---
        self.rag_bot.save_to_memory(
            state.model_dump(),     # Convert BaseModel → dict
            state.user_id,
            state.question,
            initial_answer
        )

        # --- Return updated state (immutably) ---
        return state.model_copy(update={
            "initial_answer": initial_answer,
            "retrieved_docs": retrieved_docs
        })

    


# from langchain.evaluation.qa import QAEvalChain

# ==============================================================================
# 1. Structured Output Schema (The Judge's Verdict Format)
# ==============================================================================

class RAGEvaluation(BaseModel):
    """The structured output format for the LLM judge's final decision."""
    
    # Faithfulness (Grounding) Check
    faithfulness_score: Literal["YES", "NO"] = Field(
        description="Must be 'YES' if the generated answer is entirely supported by the provided context. Must be 'NO' if any part of the answer is not found in the context (i.e., a hallucination)."
    )
    faithfulness_reasoning: str = Field(
        description="Detailed step-by-step reasoning for the faithfulness score. Quote the unsupported part if the score is 'NO'."
    )
    
    # Relevance Check
    relevance_score: Literal["YES", "NO"] = Field(
        description="Must be 'YES' if the generated answer directly and fully addresses the user's question. Must be 'NO' if the answer is vague, off-topic, or incomplete."
    )
    relevance_reasoning: str = Field(
        description="Detailed reasoning for the relevance score."
    )

# ==============================================================================
# 2. Custom Judge Prompt
# ==============================================================================

# This is the core 'Judge' prompt, instructing the LLM on its two tasks.
EVALUATOR_PROMPT = """
You are an expert RAG (Retrieval-Augmented Generation) evaluator. 
Your task is to critique a generated answer based on a question and a set of retrieved documents.

Evaluate on two criteria:

---
**CRITERIA 1: FAITHFULNESS (Grounding)**
* **GOAL:** Detect hallucinations.
* Every fact in the answer must be either:
  1. Explicitly stated in the context, OR
  2. Logically inferable from it (e.g., when thresholds, ranges, or hierarchical rules clearly apply).
* **SCORE:**
  - "YES" if the answer is fully supported or logically inferable.
  - "NO" only if it contradicts the context or adds unsupported information.

**CRITERIA 2: RELEVANCE**
* **GOAL:** Check whether the answer directly and sufficiently addresses the question.
* **SCORE:**
  - "YES" if the answer is clear, focused, and complete.
  - "NO" if it is off-topic, partial, or vague.

---
**INPUT DATA**
QUESTION: {question}
CONTEXT:
{context}
--------------------
GENERATED ANSWER: {answer}
--------------------

Return a JSON object strictly in this format:

{{
  "Faithfulness": "YES" or "NO",
  "Relevance": "YES" or "NO",
  "Reasoning": "Brief explanation."
}}
"""




# ==============================================================================
# 3. The Evaluator Node Class
# ==============================================================================

class EvaluatorNode:
    """
    An LLM-as-a-Judge node that evaluates RAG output based on Faithfulness and Relevance.
    """
    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0):
        # The LLM setup for the Judge: uses structured output for reliable parsing
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(RAGEvaluation)
        self.prompt = ChatPromptTemplate.from_template(EVALUATOR_PROMPT)
        self.eval_chain = self.prompt | self.structured_llm

        print(f"✅ Evaluator initialized using model: {model_name}")

    def execute(self, state: "ChatState") -> "ChatState":
        """
        Executes the evaluation chain and updates the ChatState model.
        """

        # -----------------------------
        # Extract needed fields
        # -----------------------------
        question = state.question
        answer = state.initial_answer
        retrieved_docs = state.retrieved_docs or []

        # Build context string from retrieved documents
        context = "\n\n---\n\n".join(
            [f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
        )

        try:
            # ---------------------------------------------------
            # 1. Run the LLM Judge (Evaluator Chain)
            # ---------------------------------------------------
            print("\n🧠 Running LLM Judge (Evaluating Faithfulness and Relevance)...")

            graded_output: RAGEvaluation = self.eval_chain.invoke({
                "question": question,
                "answer": answer,
                "context": context
            })

            # ---------------------------------------------------
            # 2. Parse Scores
            # ---------------------------------------------------
            faithfulness_score = 1.0 if graded_output.faithfulness_score == "YES" else 0.0
            relevance_score = 1.0 if graded_output.relevance_score == "YES" else 0.0

            # ---------------------------------------------------
            # 3. Threshold Pass = Both must be "YES"
            # ---------------------------------------------------
            threshold_passed = (faithfulness_score == 1.0 and relevance_score == 1.0)

            eval_text = (
                f"Faithfulness: {graded_output.faithfulness_score} | "
                f"Reasoning: {graded_output.faithfulness_reasoning}\n"
                f"Relevance: {graded_output.relevance_score} | "
                f"Reasoning: {graded_output.relevance_reasoning}"
            )

            print("--- Evaluation Result ---")
            print(eval_text)
            print(f"Final Decision: {'PASS' if threshold_passed else 'FAIL'}")
            print("-------------------------")

            # ---------------------------------------------------
            # 4. Update State (BaseModel → assign fields directly)
            # ---------------------------------------------------
            state.threshold_passed = threshold_passed
            state.eval_text = eval_text
            state.eval_score_faithfulness = faithfulness_score
            state.eval_score_relevance = relevance_score

        except Exception as e:
            print(f"❌ Evaluation failed due to an error: {e}")

            state.threshold_passed = False
            state.eval_text = f"Error during evaluation: {e}"
            state.eval_score_faithfulness = 0.0
            state.eval_score_relevance = 0.0

        return state



class RerankNode:
    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-large")

    def execute(self, state: "ChatState") -> "ChatState":
        question = state.question
        docs = state.retrieved_docs

        if not docs:
            return state

        pairs = [(question, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        print("🔄 Reranker scores:", scores)

        ranked_docs = [
            d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        ]

        # Update state using .copy(update={})
        return state.copy(update={
            "filtered_docs": ranked_docs[:5]
        })



class GenerationNode:
    def __init__(self, llm):
        self.llm = llm

    def execute(self, state: "ChatState") -> "ChatState":
        question = state.question
        docs = state.filtered_docs or state.retrieved_docs

        # Build context text
        context = "\n\n---\n\n".join(
            [d.page_content for d in docs]
        )

        prompt = (
            "Answer the question based ONLY on the context below.\n\n"
            f"{context}\n\n"
            f"Q: {question}\nA:"
        )

        response = self.llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        return state.copy(update={
            "answer": answer_text,
            "rag_response": answer_text,
            "final_context": context
        })


from langchain_core.language_models import BaseChatModel # Use BaseChatModel for type hinting
from langchain_core.documents import Document
class RegenerateNode:
    """
    Node for conditional re-generation of the answer.
    Includes internal Query Refinement before regeneration.
    """

    def __init__(self, llm: BaseChatModel, refiner_llm: BaseChatModel = None):
        self.llm = llm
        self.refiner_llm = refiner_llm or llm  

        self.refine_prompt = ChatPromptTemplate.from_template("""
You are a query refiner for retrieval-based QA.
Rewrite the given question so it is explicit, unambiguous, and retrieval-optimized.
Keep the meaning identical but expand shorthand or vague terms.
Return only the refined query text.

Original: {question}
Refined:
""")

    def refine_query(self, question: str) -> str:
        """Refine the user's question for better retrieval or regeneration context."""
        try:
            refined = self.refiner_llm.invoke(
                self.refine_prompt.format(question=question)
            )
            refined_text = (
                refined.content.strip()
                if hasattr(refined, "content")
                else str(refined).strip()
            )
            print(f"🪄 Refined Query: {refined_text}")
            return refined_text
        except Exception as e:
            print(f"⚠️ Query refinement failed: {e}")
            return question

    # -------------------------------------------------------------------------
    #                           UPDATED EXECUTE()
    # -------------------------------------------------------------------------
    def execute(self, state: "ChatState") -> "ChatState":
        """
        Regenerates the answer ONLY when evaluation failed.
        """

        # ------------------------------------
        # PASS CASE → No regeneration needed
        # ------------------------------------
        if state.threshold_passed:
            print("🧠 Evaluation passed. No regeneration needed.")
            state.final_answer = state.initial_answer
            return state

        print("\n🔄 Evaluation failed. Initiating Regenerate Node...")

        # ------------------------------------
        # Retrieve necessary fields
        # ------------------------------------
        original_question = state.question
        initial_answer = state.initial_answer
        eval_text = state.eval_text
        docs = state.filtered_docs or state.retrieved_docs or []

        if not docs:
            print("⚠️ Regeneration halted: No retrieved documents.")
            state.final_answer = (
                "I’m sorry, I don’t have that information. No documents were found."
            )
            return state

        # ------------------------------------
        # Build context string
        # ------------------------------------
        context = "\n\n---\n\n".join(
            [f"Source {i+1}: {d.page_content}" for i, d in enumerate(docs)]
        )

        # ------------------------------------
        # Step 1: Internal Query Refinement
        # ------------------------------------
        refined_question = self.refine_query(original_question)

        # Short rephrase step using same LLM
        rephrase_prompt = f"""
Rephrase the following question to maximize semantic alignment 
with internal policy documentation.

Question:
{original_question}
"""
        refined_q_result = self.llm.invoke(rephrase_prompt)
        refined_question = refined_q_result.content.strip()

        # ------------------------------------
        # Step 2: Build Regeneration Prompt
        # ------------------------------------
        REGEN_PROMPT_TEMPLATE = """
You are a chatbot assistant for MOL Chemical Tankers, providing precise and accurate
answers based strictly on the provided context.

Your previous answer failed evaluation. Use the critique and refined question
to correct the answer.

---
**REFINED QUESTION:** {refined_question}
**ORIGINAL QUESTION:** {original_question}

**PREVIOUS ANSWER:**
{initial_answer}

**LLM JUDGE CRITIQUE:**
{eval_text}

---
**AVAILABLE CONTEXT:**
{context}

---
**REGENERATION INSTRUCTIONS**
1. Fix all issues highlighted by the evaluation critique.
2. Remove any unsupported or hallucinated details.
3. Base the answer **only** on provided context.
4. If answer is not found, say: "I’m sorry, I don’t have that information."
5. Be clear, concise, and professional.

**NEW, CORRECTED ANSWER:**
"""

        regeneration_prompt = REGEN_PROMPT_TEMPLATE.format(
            refined_question=refined_question,
            original_question=original_question,
            initial_answer=initial_answer,
            eval_text=eval_text,
            context=context
        )

        # ------------------------------------
        # Step 3: Regenerate using LLM
        # ------------------------------------
        result = self.llm.invoke(regeneration_prompt)
        final_answer = result.content.strip()

        # ------------------------------------
        # Step 4: Update State (BaseModel)
        # ------------------------------------

        state.final_answer = final_answer
        state.refined_question = refined_question
        state.regeneration_count = (state.regeneration_count or 0) + 1

        return state
