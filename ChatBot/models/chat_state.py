from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field

class ChatState(BaseModel):
    question: str
    user_id: str
    vector_store_path: Optional[str] = None

    retrieved_docs: List[Any] = Field(default_factory=list)
    filtered_docs: List[Any] = Field(default_factory=list)
    final_context: List[Any] = Field(default_factory=list)

    initial_answer: str = ""
    final_answer: str = ""

    graded_scores: List[Any] = Field(default_factory=list)
    threshold_passed: bool = False

    eval_text: str = ""
    eval_score_faithfulness: float = 0.0
    eval_score_relevance: float = 0.0

    regeneration_count: int = 0
    refined_question: Optional[str] = None

    chat_memory: Dict[str, List[tuple]] = Field(default_factory=dict)
