# self_rag_graph.py
from nodes.rewrite_node import RewriteNode
from nodes.retriever_node import RetrieverNode
from nodes.initial_llm_node import InitialLLMNode
from nodes.evaluator_node import EvaluatorNode
from nodes.refine_query_node import RefineQueryNode
from nodes.regenerate_node import RegenerateNode
from nodes.answer_node import AnswerNode

MAX_REGENERATIONS = 3

class SelfRAGRunner:
    def __init__(self, rag):
        self.rag = rag
        # instantiate nodes
        self.rewrite_node = RewriteNode(rag)
        self.retriever_node = RetrieverNode(rag)
        self.initial_llm_node = InitialLLMNode(rag)
        self.evaluator_node = EvaluatorNode(rag)
        self.refine_query_node = RefineQueryNode(rag)
        self.regenerate_node = RegenerateNode(rag)
        self.answer_node = AnswerNode()

    def run(self, user_id: str, question: str) -> dict:
        """
        Executes the Self-RAG pipeline and returns final state dict which includes final_answer.
        """
        
        # shared state dict (keys names match what nodes expect)
        state = {
            "user_id": user_id,
            "question": question,
            "refined_question": None,
            "retrieved_docs": [],
            "draft_answer": None,
            "eval_text": None,
            "eval_score_faithfulness": 0.0,
            "eval_score_relevance": 0.0,
            "threshold_passed": False,
            "regeneration_count": 0,
            "final_answer": None
        }
               # 0) Check for small talk first (using existing logic from rag)
        is_small_talk = self.rag._is_small_talk(question)
        if is_small_talk:
            print(f"[SelfRAGRunner] Detected small talk: {question}")
            reply = self.rag._generate_small_talk_response(question)
            
            # Update memory with small talk interaction
            if user_id not in self.rag.chat_memory:
                from collections import deque
                self.rag.chat_memory[user_id] = deque(maxlen=self.rag.history_limit)
            
            q_record = {
                "original": question,
                "rewritten": None,
                "history_used": ""
            }
            self.rag.chat_memory[user_id].append((q_record, reply))
            
            # Return small talk response as final answer
            state["final_answer"] = reply
            return state
        # 1) Rewrite
        out = self.rewrite_node.execute(state)
        state.update(out)

        # 2) Retrieve
        out = self.retriever_node.execute(state)
        state.update(out)

        # 3) Generate initial draft
        out = self.initial_llm_node.execute(state)
        state.update(out)

        # 4) Evaluate + possible loop
        while True:
            out = self.evaluator_node.execute(state)
            state.update(out)
            if state.get("threshold_passed"):
                break

            # if failed and regeneration budget available
            if state.get("regeneration_count", 0) >= MAX_REGENERATIONS:
                # give up and keep current draft
                break

            # refine query
            out = self.refine_query_node.execute(state)
            state.update(out)

            # re-retrieve with refined query (optional: you may reuse previous docs but safer to re-retrieve)
            out = self.retriever_node.execute(state)
            state.update(out)

            # regenerate answer
            out = self.regenerate_node.execute(state)
            state.update(out)

        # 5) Final answer (only writer to final_answer)
        out = self.answer_node.execute(state)
        state.update(out)

        # Persist memory / finalize if required
        # optional: use rag.save_to_memory(...) if you want consistent storage

        return state
