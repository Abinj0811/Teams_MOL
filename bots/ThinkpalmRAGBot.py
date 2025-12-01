import asyncio
from botbuilder.schema import Activity, ActivityTypes
import os
from datetime import datetime, timedelta
from botbuilder.core import ActivityHandler, TurnContext

from graphs.main_graph import chat_graph, rag_instance
from utils.error_handling import PipelineState


# ------------------------------
# Auto session timeout settings
# ------------------------------
SESSION_TIMEOUT = timedelta(minutes=30)
user_last_seen = {}      # Maps user_id → last_active_timestamp



class ThinkpalmRAGBot(ActivityHandler):

    async def on_message_activity(self, turn_context: TurnContext):
        user_msg = turn_context.activity.text.strip()
        user_id = turn_context.activity.from_property.id
        now = datetime.utcnow()

        # --------------------------------------------------------
        # (1) Auto-persist if last session expired
        # --------------------------------------------------------
        last_seen = user_last_seen.get(user_id)
        if last_seen and (now - last_seen) > SESSION_TIMEOUT:
            rag_instance.persist_user_history(user_id)
            await turn_context.send_activity("💾 Your previous session was saved due to inactivity.")

        user_last_seen[user_id] = now

        # --------------------------------------------------------
        # (2) Explicit exit: save and quit
        # --------------------------------------------------------
        if user_msg.lower() in ["exit", "quit", "bye", "end"]:
            rag_instance.persist_user_history(user_id)
            await turn_context.send_activity("💾 Session saved. Goodbye!")
            return

        # --------------------------------------------------------
        # (3) Prepare PipelineState
        # --------------------------------------------------------
        state = PipelineState()
        state.set_state("user_id", user_id)
        state.set_state("question", user_msg)

        # Cosmos config
        state.set_state("cosmos_endpoint", os.getenv("COSMOS_ENDPOINT"))
        state.set_state("cosmos_key", os.getenv("COSMOS_KEY"))
        state.set_state("cosmos_database", os.getenv("COSMOS_DATABASE"))
        state.set_state("cosmos_container", os.getenv("COSMOS_CONTAINER"))
        state.set_state("chat_container", os.getenv("CHAT_CONTAINER"))

        state.set_state("chat_memory", {})

        # --------------------------------------------------------
        # (4) Start typing indicator immediately
        # --------------------------------------------------------
        typing_activity = Activity(type=ActivityTypes.typing)
        await turn_context.send_activity(typing_activity)

        async def send_typing_periodically():
            """Repeat typing indicator every 3 seconds."""
            try:
                while True:
                    await asyncio.sleep(3)
                    await turn_context.send_activity(typing_activity)
            except asyncio.CancelledError:
                pass

        typing_task = asyncio.create_task(send_typing_periodically())

        # --------------------------------------------------------
        # (5) Invoke LangGraph inside executor (sync call)
        # --------------------------------------------------------
        try:
            final_state = await self._invoke_graph_async(state.to_dict())
            answer = final_state.get("final_answer", "⚠️ No response generated.")
        finally:
            # stop typing indicator
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # --------------------------------------------------------
        # (6) Return answer to user
        # --------------------------------------------------------
        await turn_context.send_activity(answer)

        # Keep memory for next turn
        state.set_state("chat_memory", final_state.get("chat_memory", {}))


    async def _invoke_graph_async(self, state_dict):
        """Run graph.invoke() in executor without blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: chat_graph.invoke(state_dict))
