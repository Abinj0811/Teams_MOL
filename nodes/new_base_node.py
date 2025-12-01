from abc import ABC, abstractmethod
import logging

# Simple logger setup
pipeline_logger = logging.getLogger("PipelineLogger")
pipeline_logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
pipeline_logger.addHandler(ch)

class BaseNode(ABC):
    """Base class for all nodes in the pipeline."""

    def __init__(self, name: str):
        self.name = name
        self.logger = pipeline_logger

    @abstractmethod
    def execute(self, state: dict) -> dict:
        """
        Every node must implement this method.
        Accepts a dict as input and returns a dict as output.
        """
        pass
