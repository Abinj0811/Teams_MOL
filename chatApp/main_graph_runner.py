from utils.error_handling import PipelineState
from graphs.main_graph import chat_graph
import os 
from dotenv import load_dotenv
load_dotenv()
state = {
    # "question": str,
    # "retrieved_docs": list,        # documents from vectorstore
    # "graded_scores": list,          # confidences / relevances
    # "filtered_docs": list,         # docs that pass threshold
    # "use_web_search": bool,
    # "web_docs": list,OpenAIEmbeddings
    # "final_context": list,          # docs fed to LLM
    # "answer": str
}

state["vector_store_path"] = os.getenv("EMBEDDING_DB_NAME", "./app/vectorstores/faiss_index/MOL_openai_H")
state["question"] = "I want to submit an approval to implement a new software system where 40% of the cost to implement and maintain will be charged to UNIX. Amount for implementation is $40,000 ($24,000 - MCT, $16,000 - UNIX) and maintenance is $30,000 ($18,000 - MCT, $12,000 - UNIX). What approval criteria should I use, and is it required to obtain separate subsidiary approval?."

final_state = chat_graph.invoke(state)

print("✅ Chat complete.")
print("🧠 RAG Response:\n", final_state.get("rag_response"))
