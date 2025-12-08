# nodes/initial_llm_node.py
from nodes.base_node import BaseNode

class InitialLLMNode(BaseNode):
    def __init__(self, rag):
        super().__init__("InitialLLMNode")
        self.rag = rag

    def execute(self, state: dict) -> dict:
        """
        Produces a draft answer using the prompt template + retrieved docs + refined_question.
        Writes only: draft_answer
        """
        refined_q = state.get("refined_question", "").strip()
        docs = state.get("retrieved_docs", [])

        # Build prompt using rag._build_prompt_template
        # The template expects {question} and {context} – we'll render a plain string prompt
        prompt_template = self.rag._build_prompt_template(refined_q)
        context_text = "\n\n".join([self.rag.clean_content(d) for d in docs])
        formatted = prompt_template.format_prompt(
            question=refined_q,
            context=context_text
        )
        prompt_text = formatted.to_string()


        # Call LLM to get draft
        llm_resp = self.rag.llm.invoke(prompt_text).content.strip()

        return {
            "draft_answer": llm_resp
        }
