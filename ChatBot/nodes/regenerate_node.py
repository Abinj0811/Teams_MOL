# nodes/regenerate_node.py
from nodes.base_node import BaseNode

class RegenerateNode(BaseNode):
    def __init__(self, rag):
        super().__init__("RegenerateNode")
        self.rag = rag

    def execute(self, state: dict) -> dict:
        """
        Re-generate a draft answer after refinement.
        Writes only: draft_answer
        """
        refined_q = state.get("refined_question", "")
        docs = state.get("retrieved_docs", [])
        prompt_template = self.rag._build_prompt_template(refined_q)
        context_text = "\n\n".join([self.rag.clean_content(d) for d in docs])
        # Correct LangChain formatting for ChatPromptTemplate
        formatted_prompt = prompt_template.format_prompt(
            question=refined_q,
            context=context_text
        )

        prompt_text = formatted_prompt.to_string()
        llm_resp = self.rag.llm.invoke(prompt_text).content.strip()
        return {
            "draft_answer": llm_resp
        }
