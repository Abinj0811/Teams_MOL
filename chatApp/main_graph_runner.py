import os
from dotenv import load_dotenv
from graphs.main_graph import chat_graph, rag_instance  # your existing graph pipeline
from utils.error_handling import PipelineState

load_dotenv()

# """
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
    state.set_state("chat_memory", {})  # 🧠 initialize empty memory

    # try:
    # Start chat loop
    # while True:
    # '''
    msg_list = ["Who are the people involved in the Ship Management Committee?"]
    msg_list=[ 
            "who can approve new product acquisition of less than US$50,000 ",
            "Do we need to apply application of approval to write off the golf membership cost that was migrated from FCCSP (about $25,000)?",
                "I want to submit an approval application for the service agreement with external party to implement a new expense claim system that cost US$50,000 and 1 year maintenance service contract that costs USD85,000. What approval criteria should I use?",
                "I want to submit an approval to implement a new software system where 40% of the cost to implement and maintain will be charged to UNIX. Amount for implementation is $40,000 ($24,000 - MCT, $16,000 - UNIX) and maintenance is $30,000 ($18,000 - MCT, $12,000 - UNIX). What approval criteria should I use, and is it required to obtain separate subsidiary approval?"
                ]
    
    msg_list= ["Novation of the Time Charter contract with XXX Company."]
    msg_list= [
        "Approval criteria and type for P&I Insurance (CLI/FDD) for Policy Year 2025",
                "Which dept is responsible for vessels-related insurances (TCL, DTH, FDD)?"
                ]
    msg_list= ["What approval and departments are involved for the conclusion of service agreement with MCTWTN for admin cost sharing of USD800,000?"]

    msg_list= [
        # "Does A4 approval requires submission of approval application?",
        #         "What approval criteria should be applied for JOL contract time charter for 5 years period?",
    #         # "What approval and departments are involved for the revision of service agreement with Unix for office cost sharing of USD200,000?",
    # "Who are the people involved in the Ship Management Committee?",
    # "what is its responsibilities",
    "what is official authority regulations",
    "what is its objective"
    ]
    
    # msg_list = []
    
    for user_msg in msg_list:
        # '''
        # user_msg = input("\nYou: ")
        if user_msg.lower() in ["exit", "quit"]:
            print("\n💾 sample History saved and exiting.")
            rag_instance.persist_user_history(user_id, state)
            break
        
        

        # Update state for this turn
        state.set_state("question", user_msg)

        # Invoke your RAG graph
        final_state = chat_graph.invoke(state.to_dict())
        state.set_state("chat_memory", final_state["chat_memory"])
        # Extract the RAG answer
        rag_answer = final_state.get("rag_response", "⚠️ No response generated.")
        
        related_docs = final_state.get("retrieved_docs", [])

        # print(f"Assistant: {rag_answer}\n")

        if related_docs:
            # print(related_docs)
            print(f"📚 Related documents: {[d.metadata.get('doc_id', 'unknown') for d in related_docs]}\n")
    rag_instance.persist_user_history(user_id, state)
    # except Exception  as e:
    #     print('Error: ', e)
        # rag_instance.persist_user_history(user_id, state)
    #     print("\n💾 Sample History saved and exiting.")
"""  
        
from nodes.chat_node import ThinkpalmRAG
if __name__ == "__main__":
    user_id = "test_user"

    rag = ThinkpalmRAG(
        # os.getenv("COSMOS_ENDPOINT"),
        # os.getenv("COSMOS_KEY"),
        # os.getenv("COSMOS_DATABASE"),
        # os.getenv("COSMOS_CONTAINER"),
        # os.getenv("CHAT_CONTAINER")
    )

    choice = input("Do you want to clear chat history for this user? (y/n): ")
    if choice.lower().startswith("y"):
        rag.clear_user_history(user_id)
        exit()
# """