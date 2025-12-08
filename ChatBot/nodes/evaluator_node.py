# nodes/evaluator_node.py
from nodes.base_node import BaseNode
import json

class EvaluatorNode(BaseNode):
    def __init__(self, rag, threshold=0.75):
        super().__init__("EvaluatorNode")
        self.rag = rag
        self.threshold = threshold

    def execute(self, state: dict) -> dict:
        draft = state.get("draft_answer", "")
        question = state.get("refined_question", "")
        docs = state.get("retrieved_docs", [])

        context_text = "\n\n".join([self.rag.clean_content(d) for d in docs])

        eval_prompt = f"""
Evaluate the LLM DRAFT ANSWER using ONLY the context below.

Question: {question}
Draft Answer: {draft}
Context: {context_text}

Return JSON:
{{ "faithfulness": <number 0-1>, "relevance": <number 0-1>, "feedback": "<short text>" }}
"""
        llm_resp = self.rag.llm.invoke(eval_prompt).content
        try:
            result = json.loads(llm_resp)
        except Exception:
            # If LLM didn't return strict JSON, fall back to conservative fail
            result = {"faithfulness": 0.0, "relevance": 0.0, "feedback": "Could not parse evaluation output."}

        passed = (result.get("faithfulness", 0) >= self.threshold and
                  result.get("relevance", 0) >= self.threshold)

        return {
            "eval_text": result.get("feedback", ""),
            "eval_score_faithfulness": float(result.get("faithfulness", 0)),
            "eval_score_relevance": float(result.get("relevance", 0)),
            "threshold_passed": passed
        }
