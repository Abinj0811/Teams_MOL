from nodes.new_base_node import BaseNode
from utils.error_handling import PipelineState
from chains_old.embedding_chain import FAISSEmbeddingHandler  # your class

class FAISSEmbeddingNode(BaseNode):
    """Node to create FAISS embeddings for all markdown documents."""

    def __init__(self):
        super().__init__("FAISSEmbeddingNode")
        self.handler = FAISSEmbeddingHandler()

    def validate_input(self, state: PipelineState) -> bool:
        md_files = state.get("markdown_documents")
        if not md_files:
            self.logger.log_warning(self.name, "No markdown documents found for embedding.")
            return False
        return True

    def execute(self, state: PipelineState) -> PipelineState:
        md_files = state.get("processed_results", [])
        saved_indices = []
        print("md_files:", md_files)
        for file_path in md_files:
            try:
                self.logger.info(f"[{self.name}] Creating embeddings for {file_path}")
                self.handler.create_faiss_index(file_path)
                saved_indices.append(self.handler.db_path)
            except Exception as e:
                self.logger.log_error(self.name, e, f"Embedding failed for {file_path}")

        state["vectorstores"] = saved_indices
        self.logger.info(f"[{self.name}] Created {len(saved_indices)} FAISS vectorstores")
        return state
