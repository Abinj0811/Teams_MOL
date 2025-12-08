from graph.self_rag_graph import SelfRAGRunner
from models.chat_state import ChatState
from rag.thinkpalm_rag import ThinkpalmRAG
import os
from utils.error_handling import PipelineState

def ask_question(user_id, question):
    state = ChatState(
        user_id=user_id,
        question=question
    )
    rag = ThinkpalmRAG()
    runner = SelfRAGRunner(rag)
    return rag
 
if __name__ == "__main__":
    user_id = "test_user_123"
    # question = input("Ask a question: ")
    # user_msg = "Novation of the Time Charter contract with XXX Company."
    user_msg = "who can approve new product acquisition of less than US$20,000"
    user_msg = "what is the aproval criteria for Conclusion of service agreement with MCTSPR subsidiaries if the amount is USD 680,000"
    rag = ask_question(user_id, user_msg)
    if user_msg.lower() in ["exit", "quit", "bye", "end"]:
        rag.persist_user_history(user_id)


    # ============================================================
    # (4) BUILD PIPELINE STATE FOR THIS TURN
    # ================="feedback": "The draft answer is mostly relevant but not fully faithful to the context. For the conclusion of a service agreement with MCTSPR subsidiaries for USD 680,000, the correct approval criteria per the context (Authority Table, Page 15) is: Authorised Approvers: A1, Deliberation by MM, Review to GPM, Co-Management Dept: Relevant Dept. The draft incorrectly states A2 as the approver and omits the requirement for deliberation by MM. The MM report requirement is correct, but the answer should specify 'Deliberation by MM' for amounts over USD 500,000."===========================================
    state = PipelineState()
    state.set_state("user_id", user_id)
    state.set_state("question", user_msg)

    # Cosmos config
    state.set_state("cosmos_endpoint", os.getenv("COSMOS_ENDPOINT"))
    state.set_state("cosmos_key", os.getenv("COSMOS_KEY"))
    state.set_state("cosmos_database", os.getenv("COSMOS_DATABASE"))
    state.set_state("ENRICHED_CONTAINER", os.getenv("ENRICHED_CONTAINER"))
    state.set_state("chat_container", os.getenv("CHAT_CONTAINER"))

    # graph fills this
    state.set_state("chat_memory", {})
    
    runner = SelfRAGRunner(rag)
    final_state = runner.run(user_id, user_msg)
    answer = final_state.get("final_answer")
 
    print(answer)