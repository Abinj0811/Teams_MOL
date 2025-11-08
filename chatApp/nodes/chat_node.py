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
from collections import deque
from azure.cosmos import CosmosClient, exceptions, PartitionKey
import json
import uuid
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
        self.chat_memory = {}  # { user_id: deque([(user_msg, assistant_msg), ...]) }
        self.last_sync_counter = {}   # track per-user unsynced turns
        self.history_limit = 5
        self.autosave_interval = 3  
        

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
            exit()
            # Delete each item
            for item in items:
                item_id = item["id"]
                self.history_container.delete_item(item=item_id, partition_key=user_id)

            print(f"✅ Cleared all chat history for user '{user_id}'.")
            return True

        except Exception as e:
            print(f"❌ Error clearing history for {user_id}: {e}")
            return False


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
        context_docs = inputs.get("context_docs", [])
        formatted_context = "\n\n".join([
            d.page_content if hasattr(d, "page_content") else d.get("page_content", "")
            for d in context_docs
        ])

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

        sql = f"""
        SELECT TOP {self.top_k}
            c.id, c.text, c.metadata,
            VectorDistance(c.vector_embedding, {emb_json}) AS score
        FROM c
        ORDER BY VectorDistance(c.vector_embedding, {emb_json})
        """
        items = list(self.container.query_items(query=sql, enable_cross_partition_query=True))

        # ✅ Return proper Document objects
        return [
            Document(page_content=i.get("text", ""), metadata={"id": i.get("id"), "score": i.get("score")})
            for i in items
        ]

    # ------------------------------------------------------------
    # PROMPT + RAG CHAIN
    # ------------------------------------------------------------
    @staticmethod
    def format_docs(docs):
        """Combine retrieved docs into one context block."""
        return "\n\n".join([doc["page_content"] for doc in docs])

    def _build_prompt_template(self):
        """Create a prompt that includes chat history + context."""
        return ChatPromptTemplate.from_template("""
    You are a policy assistant for MOL Chemical Tankers.
    Answer precisely and only based on the provided context and the past conversation.

    Guidelines:
    1. Use the conversation history below to understand the topic and follow-up questions.
    2. If the question refers to "it", "this", or "explain again", look at the last assistant response in the history.
    3. Only use the provided context for factual information — do not invent details.

    ---
    ### Conversation History
    {history}

    ### Current Question
    {question}

    ### Context
    {context}

    ---
    Answer:
    """)

    
    def _build_rag_chain(self):
        """Create the retrieval + generation chain with selective memory-based rewriting."""
        prompt = self._build_prompt_template()

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
            docs = self.search_cosmos_documents(rewritten["question"])
            return {
                "context_docs": docs,
                "user_id": rewritten["user_id"],
                "question": rewritten["question"],
            }


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
        response = self.rag_chain.invoke(inputs)
        
        docs = self.search_cosmos_documents(question)
        self.update_chat_memory(user_id, question, response)
        
        return response, docs
    


class RetrieverNode(BaseNode):
    def __init__(self):
        super().__init__("RetrieverNode")
        self.rag_bot = ThinkpalmRAG()
        
    def execute(self, state: dict):
        
        state["initial_answer"], state["retrieved_docs"] = self.rag_bot.ask(state["user_id"], state["question"])
        state["rag_response"]  = state["initial_answer"]
        # ✅ Save to memory (state + class)
        self.rag_bot.save_to_memory(
            state,
            state["user_id"],
            state["question"],
            state["initial_answer"]
        )

        # docs = self.rag_bot.vector_store.similarity_search(state["question"], k=7)        
        
        return state
    
"""
# evaluation with cross encoder from sentance transformers
class EvaluatorNode:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def execute(self, state: dict):
        ans_emb = self.model.encode(state["initial_answer"])
        ctx_embs = self.model.encode([d.page_content for d in state["retrieved_docs"]])
        # Compute cosine similarities
        sims = [
            float(np.dot(ans_emb, e) / (np.linalg.norm(ans_emb) * np.linalg.norm(e)))
            for e in ctx_embs
        ]
        avg_score = float(np.mean(sims))

        print(f"🧩 Relevance Score: {avg_score:.3f}")

        # Save results into state
        state["scores"] = sims
        state["avg_score"] = avg_score
        state["threshold_passed"] = avg_score > 0.6
        return state
"""

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
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        # The LLM setup for the Judge: uses structured output for reliable parsing
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(RAGEvaluation)
        self.prompt = ChatPromptTemplate.from_template(EVALUATOR_PROMPT)
        self.eval_chain = self.prompt | self.structured_llm

        print(f"✅ Evaluator initialized using model: {model_name}")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the evaluation chain and updates the state.
        
        Expected State Keys:
        - "question": str
        - "initial_answer": str
        - "retrieved_docs": List[Document]
        """
        question = state["question"]
        answer = state["initial_answer"]
        # Concatenate retrieved document content into a single context string
        context = "\n\n---\n\n".join(
            [f"Source {i+1}: {d.page_content}" for i, d in enumerate(state.get("retrieved_docs", []))]
        )
        # with open("debug_context.txt", "w") as f:
        #     f.write(context)
        
        try:
            # 1. Run the LLM-as-a-Judge Chain
            print("\n🧠 Running LLM Judge (Evaluating Faithfulness and Relevance)...")
            graded_output: RAGEvaluation = self.eval_chain.invoke({
                "question": question,
                "answer": answer,
                "context": context
            })
            
            # 2. Process Scores
            faithfulness_score = 1.0 if graded_output.faithfulness_score == "YES" else 0.0
            relevance_score = 1.0 if graded_output.relevance_score == "YES" else 0.0
            
            # 3. Determine Overall Pass/Fail (Threshold Check)
            # Both metrics must pass for the answer to be considered good.
            overall_pass = (faithfulness_score == 1.0) and (relevance_score == 1.0)
            
            # 4. Compile summary for debugging/logging
            eval_text = (
                f"Faithfulness: {graded_output.faithfulness_score} | Reasoning: {graded_output.faithfulness_reasoning}\n"
                f"Relevance: {graded_output.relevance_score} | Reasoning: {graded_output.relevance_reasoning}"
            )
            
            print("--- Evaluation Result ---")
            print(eval_text)
            print(f"Final Decision: {'PASS' if overall_pass else 'FAIL'}")
            print("-------------------------")

            # 5. Update State
            state["threshold_passed"] = overall_pass
            state["eval_text"] = eval_text
            state["eval_score_faithfulness"] = faithfulness_score
            state["eval_score_relevance"] = relevance_score

        except Exception as e:
            print(f"❌ Evaluation failed due to an error: {e}")
            state["threshold_passed"] = False
            state["eval_text"] = f"Error during evaluation: {e}"
            state["eval_score_faithfulness"] = 0.0
            state["eval_score_relevance"] = 0.0
            
        return state


class RerankNode:
    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-large")

    def execute(self, state: dict):
        question = state["question"]
        docs = state["retrieved_docs"]
        pairs = [(question, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        print("from reranked node", scores)
        ranked_docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        state["filtered_docs"] = ranked_docs[:5]  # top 5
        return state


class GenerationNode:
    def __init__(self, llm):
        self.llm = llm

    def execute(self, state: dict) -> dict:
        context = state.get("final_context", [])
        question = state.get("question", "")
        # Create prompt
        prompt = f"Answer the question based only on the context below:\n\n{context}\n\nQ: {question}\nA:"
        # Use LLM to generate
        response = self.llm.invoke(prompt)
        state["answer"] = response.content if hasattr(response, "content") else str(response)
        state["rag_response"] = state["answer"]
        
        
        return state
    


from langchain_core.language_models import BaseChatModel # Use BaseChatModel for type hinting
from langchain_core.documents import Document

class RegenerateNode:
    """
    A node for conditional re-generation of the answer. 
    It leverages the LLM Judge's evaluation feedback to improve the answer.
    Now includes internal Query Refinement before regeneration.
    """

    def __init__(self, llm: BaseChatModel, refiner_llm: BaseChatModel = None):
        self.llm = llm
        # Lightweight refiner model (can reuse llm if not given)
        self.refiner_llm = refiner_llm or llm  

        # Define the query refiner prompt
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
            refined = self.refiner_llm.invoke(self.refine_prompt.format(question=question))
            refined_text = refined.content.strip() if hasattr(refined, "content") else str(refined).strip()
            print(f"🪄 Refined Query: {refined_text}")
            return refined_text
        except Exception as e:
            print(f"⚠️ Query refinement failed: {e}")
            return question  # fallback to original

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conditionally regenerates the answer using the judge's feedback if evaluation fails.
        Now refines the query internally before re-generation.
        """
        # If evaluation passed, skip regeneration
        if state.get("threshold_passed", False):
            print("🧠 Evaluation passed. No regeneration needed.")
            return {**state, "final_answer": state["initial_answer"]}

        print("\n🔄 Evaluation failed. Initiating Regenerate Node...")

        # Retrieve documents
        docs: List[Document] = state.get("filtered_docs") or state.get("retrieved_docs") or []
        if not docs:
            final_answer = "I’m sorry, I don’t have that information. No relevant documents were retrieved for this query."
            print("⚠️ Regeneration halted: No documents available for context.")
            return {**state, "final_answer": final_answer}

        # 🔍 Refine the query before using it in regeneration
        original_question = state["question"]
        refined_question = self.refine_query(original_question)

        context = "\n\n---\n\n".join([f"Source {i+1}: {d.page_content}" for i, d in enumerate(docs)])
        initial_answer = state["initial_answer"]
        eval_text = state.get("eval_text", "Evaluation feedback was unavailable.")

        # Build improved regeneration prompt
        # 4. Base your answer strictly on the provided context or semantically related items within it.If the answer is not available, respond with: "I’m sorry, I don’t have that information."

        rephrase_prompt = f"""
Rephrase the following question to maximize semantic match 
with company policy and approval documents. 
Include equivalent or related terms (e.g., synonyms, abbreviations, departmental terms).

Question:
{original_question}
"""
        rephrase_result = self.llm.invoke(rephrase_prompt)
        refined_question = rephrase_result.content.strip()
        REGEN_PROMPT_TEMPLATE = """
You are a chatbot assistant for MOL Chemical Tankers, designed to provide precise and accurate answers based solely on the provided context. 

Your previous answer failed evaluation. Use the critique and refined query to correct it.

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

**INSTRUCTIONS FOR RE-GENERATION:**
1. Address all issues mentioned in the critique.
2. Remove unsupported or irrelevant content.
3. Your answer must be based on the provided context only.
4. Be concise, accurate, and professional.

**NEW, CORRECTED ANSWER:**
"""

        regeneration_prompt = REGEN_PROMPT_TEMPLATE.format(
        refined_question=refined_question,
        original_question=original_question,
        initial_answer=initial_answer,
        eval_text=eval_text,
        context=context
    )

        answer_result = self.llm.invoke(regeneration_prompt)

        new_answer = answer_result.content.strip()
        state["rag_response"] = new_answer 
        state["regeneration_count"] = state.get("regeneration_count", 0) + 1 
        state["refined_question"] = refined_question # store for transparency return state
        
        # ✅ Save to memory (state + class)
        # self.rag_bot.save_to_memory(
        #     state,
        #     state["user_id"],
        #     state["question"],
        #     state["initial_answer"]
        # )
