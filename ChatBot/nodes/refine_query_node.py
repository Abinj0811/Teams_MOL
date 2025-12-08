# nodes/refine_query_node.py
from nodes.base_node import BaseNode

class RefineQueryNode(BaseNode):
    def __init__(self, rag):
        super().__init__("RefineQueryNode")
        self.rag = rag

    def execute(self, state: dict) -> dict:
        feedback = state.get("eval_text", "")
        original = state.get("refined_question", "")
        refine_prompt = f"""
The previous draft was judged to be insufficient.

Original Question:
{original}

Evaluation Feedback:
{feedback}

Rewrite the question to make it more explicit and easier to retrieve relevant policy text.
Output the rewritten question on a single line.
"""
        new_q = self.rag.llm.invoke(refine_prompt).content.strip()

        regen_count = state.get("regeneration_count", 0)
        return {
            "refined_question": new_q,
            "regeneration_count": regen_count + 1
        }
