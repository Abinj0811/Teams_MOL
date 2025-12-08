# nodes/rewrite_node.py
from nodes.base_node import BaseNode

class RewriteNode(BaseNode):
    def __init__(self, rag):
        super().__init__("RewriteNode")
        self.rag = rag

    def execute(self, state: dict) -> dict:
        """
        Produce a self-contained refined question using history (if any).
        DOES NOT trigger retrieval or the RAG chain.
        Writes only: refined_question
        """
        user_id = state.get("user_id")
        question = state.get("question", "").strip()

        history_text = self.rag.get_history_for_rewrite(user_id, turns=1)
        # If no history or no pronouns, keep original question
        if not history_text:
            rewritten = question
        else:
            PRONOUN_PATTERN = self.rag.PRONOUN_PATTERN if hasattr(self.rag, "PRONOUN_PATTERN") else None
            if PRONOUN_PATTERN and PRONOUN_PATTERN.search(question):
                # Use the rag.llm to rewrite into self-contained question
                reform_prompt = f"""
You are a helper that rewrites an ambiguous user question (with pronouns) into a self-contained question using the conversation history.

Conversation History:
{history_text}

Latest User Question:
{question}

Rewritten question (single line):
"""
                rewritten = self.rag.llm.invoke(reform_prompt).content.strip()
            else:
                rewritten = question

        return {"refined_question": rewritten}
