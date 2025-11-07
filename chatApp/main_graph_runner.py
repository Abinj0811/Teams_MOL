import os
from dotenv import load_dotenv
from graphs.main_graph import chat_graph  # your existing graph pipeline
from utils.error_handling import PipelineState

load_dotenv()

if __name__ == "__main__":
    # Environment setup
    COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
    COSMOS_KEY = os.getenv("COSMOS_KEY")
    COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
    COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")
    CHAT_CONTAINER = os.getenv("CHAT_CONTAINER")

    print(f"✅ Connected to Cosmos DB: {COSMOS_DATABASE}/{COSMOS_CONTAINER}")
    
    # User session
    user_id = "test_user"
    print(f"🧠 Starting chat for user: {user_id}")
    print("Type 'exit' to stop chatting.\n")

    # Initialize a reusable graph state
    state = PipelineState()
    state.set_state("user_id", user_id)
    state.set_state("cosmos_endpoint", COSMOS_ENDPOINT)
    state.set_state("cosmos_key", COSMOS_ENDPOINT)
    state.set_state("cosmos_database", COSMOS_DATABASE)
    state.set_state("cosmos_container", COSMOS_CONTAINER)
    state.set_state("chat_container", CHAT_CONTAINER)


    # Start chat loop
    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ["exit", "quit"]:
            print("👋 Ending chat...")
            break

        # Update state for this turn
        state.set_state("question", user_msg)

        # Invoke your RAG graph
        final_state = chat_graph.invoke(state.to_dict())

        # Extract the RAG answer
        rag_answer = final_state.get("rag_response", "⚠️ No response generated.")
        related_docs = final_state.get("retrieved_docs", [])

        print(f"Assistant: {rag_answer}\n")

        if related_docs:
            print(f"📚 Related documents: {[d.get('id', 'unknown') for d in related_docs]}\n")
