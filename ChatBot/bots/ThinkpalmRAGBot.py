import asyncio
from botbuilder.schema import  ChannelAccount, ActivityTypes, Activity, Attachment
import os
from datetime import datetime, timedelta
from botbuilder.core import ActivityHandler, TurnContext

from utils.error_handling import PipelineState
from rag.thinkpalm_rag import ThinkpalmRAG
from azure.cosmos import CosmosClient, PartitionKey
from graph.self_rag_graph import SelfRAGRunner

# ------------------------------
# Auto session timeout settings
# ------------------------------
SESSION_TIMEOUT = timedelta(minutes=30)
user_last_seen = {}      # Maps user_id → last_active_timestamp


def build_answer_card(answer_text):
    return {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "TextBlock",
                "text": answer_text,
                "wrap": True,
                "markdown": True,
                "spacing": "medium"
            },
            {
                "type": "TextBlock",
                "text": "Was this response helpful?",
                "weight": "Bolder",
                "size": "small",
                "wrap": True,
                "spacing": "large",
                "separator": True
            }
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "👍",
                "data": {
                    "feedback": "thumbs_up"
                }

            },
            {
                "type": "Action.Submit",
                "title": "👎",
                "data": {
                    "feedback": "thumbs_down"
                }

                            }
        ]

    }

class ThinkpalmRAGBot(ActivityHandler):
    def __init__(self):
        # COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
        # COSMOS_KEY = os.getenv("COSMOS_KEY")
        # COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
        # COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")
        # CHAT_CONTAINER = os.getenv("CHAT_CONTAINER")

        self.rag = ThinkpalmRAG()

        self.feedback_container = self.rag.db.create_container_if_not_exists(
            id="UserFeedback",
            partition_key=PartitionKey(path="/user_id"),
            # offer_throughput=400
        )
        print("self.feedback_container")                            # <<< ADD THIS
        self.last_messages = {}
    def save_feedback(self, user, feedback, question=None, answer=None):
        print(f"[FEEDBACK] {user} = {feedback}")

        item = {
            "id": f"fb-{user}-{datetime.utcnow().isoformat()}",
            "user_id": user,
            "feedback": feedback,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            self.feedback_container.upsert_item(item)   # now using pre-initialized container
            print("✓ Feedback saved to Cosmos DB")
        except Exception as e:
            print("❌ Error saving feedback:", e)



    async def on_message_activity(self, turn_context: TurnContext):
        activity = turn_context.activity
        user_id = activity.from_property.id
        now = datetime.utcnow()

        print("activity text:", activity.text)

        # ============================================================
        # (0) SAFELY EXTRACT USER TEXT
        # Sometimes activity.text is None (thumbs reaction, etc.)
        # ============================================================
        user_msg = activity.text.strip() if activity.text else ""

        # ============================================================
        # (1) HANDLE NEGATIVE FEEDBACK BEFORE ANY RAG EXECUTION
        # ============================================================
        if activity.value and "feedback" in activity.value:
            feedback_type = activity.value["feedback"]     # like "negative"
            last = self.last_messages.get(user_id, {})

            question = last.get("question")
            answer = last.get("answer")

            self.save_feedback(user_id, feedback_type, question, answer)

            # Silent acknowledgment (optional)
            await turn_context.send_activity("🙏 Thanks for your feedback! Logged.")
            return

        # ============================================================
        # (2) INACTIVITY TIMEOUT → auto-save session
        # ============================================================
        last_seen = user_last_seen.get(user_id)
        if last_seen and (now - last_seen) > SESSION_TIMEOUT:
            self.rag.persist_user_history(user_id)
            await turn_context.send_activity("💾 Your previous session was saved due to inactivity.")

        user_last_seen[user_id] = now

        # ============================================================
        # (3) EXIT COMMANDS → persist + exit
        # ============================================================
        if user_msg.lower() in ["exit", "quit", "bye", "end"]:
            self.rag.persist_user_history(user_id)
            await turn_context.send_activity("💾 Session saved. Goodbye!")
            return

        # If message is empty (like just a reaction), ignore
        if not user_msg:
            return

        # ============================================================
        # (4) BUILD PIPELINE STATE FOR THIS TURN
        # ============================================================
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

        # ============================================================
        # (5) SEND TYPING INDICATOR
        # ============================================================
        typing_activity = Activity(type=ActivityTypes.typing)
        await turn_context.send_activity(typing_activity)

        async def periodic_typing():
            try:
                while True:
                    await asyncio.sleep(3)
                    await turn_context.send_activity(typing_activity)
            except asyncio.CancelledError:
                pass

        typing_task = asyncio.create_task(periodic_typing())

        # ============================================================
        # (6) RUN LangGraph (sync → async wrapper)
        # ============================================================
        try:
            runner = SelfRAGRunner(self.rag)
            final_state = runner.run(user_id, user_msg)
            answer = final_state.get("final_answer")

        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # ============================================================
        # (7) ADAPTIVE CARD RESPONSE
        # ============================================================
        card_json = build_answer_card(answer)

        attachment = Attachment(
            content_type="application/vnd.microsoft.card.adaptive",
            content=card_json
        )

        await turn_context.send_activity(
            Activity(
                type=ActivityTypes.message,
                attachments=[attachment]
            )
        )

        # ============================================================
        # (8) SAVE LAST Q/A FOR FUTURE FEEDBACK
        # ============================================================
        self.last_messages[user_id] = {
            "question": user_msg,
            "answer": answer
        }

        # Keep memory for next turn
        state.set_state("chat_memory", final_state.get("chat_memory", {}))



    async def _invoke_graph_async(self, state_dict):
        """Run graph.invoke() in executor without blocking."""
        # This method is not currently used, but kept for potential future use
        # If needed, create a SelfRAGRunner instance and call run() instead
        loop = asyncio.get_running_loop()
        runner = SelfRAGRunner(self.rag)
        return await loop.run_in_executor(None, lambda: runner.run(state_dict.get("user_id"), state_dict.get("question")))