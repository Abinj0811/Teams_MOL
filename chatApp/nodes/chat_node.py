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

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

class RetrieverNode(BaseNode):
    def __init__(self):
        super().__init__("RetrieverNode")
        self.rag_bot = ThinkpalmRAG()
        
    def execute(self, state: dict):
        
        state["initial_answer"], state["retrieved_docs"] = self.rag_bot.ask(state["question"])
        state["rag_response"]  = state["initial_answer"]
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
