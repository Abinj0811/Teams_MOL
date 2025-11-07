import logging
import sys
from datetime import datetime
from typing import Dict, Any


from utils.logging_config import pipeline_logger
from typing import Dict, Any, Optional
from .logging_config import pipeline_logger
import time

class NodeExecutionError(Exception):
    """Custom exception for node execution errors"""
    def __init__(self, node_name: str, message: str, original_error: Optional[Exception] = None):
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}': {message}")



class PipelineState:
    """Manages pipeline state and error handling"""
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self.warnings: Dict[str, str] = {}
    
    def set_state(self, key: str, value: Any):
        """Set state value"""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None):
        """Get state value"""
        return self.state.get(key, default)
    
    def add_error(self, node_name: str, error: str):
        """Add error to state"""
        self.errors[node_name] = error
        pipeline_logger.log_error(node_name, Exception(error))
    
    def add_warning(self, node_name: str, warning: str):
        """Add warning to state"""
        self.warnings[node_name] = warning
        pipeline_logger.log_warning(node_name, warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> str:
        """Get summary of all errors"""
        if not self.errors:
            return "No errors"
        return "; ".join([f"{node}: {error}" for node, error in self.errors.items()])

def safe_execute(func):
    """Decorator for safe node execution with error handling"""
    def wrapper(self, state: PipelineState, *args, **kwargs):
        try:
            pipeline_logger.log_node_execution(self.__class__.__name__, "STARTED")
            start_time = time.time()
            
            result = func(self, state, *args, **kwargs)
            
            execution_time = time.time() - start_time
            pipeline_logger.log_node_execution(
                self.__class__.__name__, "COMPLETED", execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            pipeline_logger.log_error(self.__class__.__name__, e)
            state.add_error(self.__class__.__name__, str(e))
            raise NodeExecutionError(self.__class__.__name__, str(e), e)
    
    return wrapper