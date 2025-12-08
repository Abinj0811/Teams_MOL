import logging
import sys
from typing import Dict, Any
import os

class PipelineLogger:
    """Centralized logging configuration for LangGraph pipeline"""
    
    def __init__(self, name: str = "langgraph_pipeline"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
    
    def log_node_execution(self, node_name: str, status: str, 
                          execution_time: float = None, 
                          additional_info: Dict[str, Any] = None):
        """Log node execution with timing and status"""
        message = f"Node '{node_name}' - Status: {status}"
        if execution_time:
            message += f" - Execution Time: {execution_time:.2f}s"
        if additional_info:
            message += f" - Info: {additional_info}"
        
        self.logger.info(message)
    
    def log_error(self, node_name: str, error: Exception, context: str = ""):
        """Log node errors with context"""
        self.logger.error(f"Node '{node_name}' - Error: {str(error)} - Context: {context}")
    
    def log_warning(self, node_name: str, message: str):
        """Log node warnings"""
        self.logger.warning(f"Node '{node_name}' - Warning: {message}")

# Global logger instance
pipeline_logger = PipelineLogger()

import csv
from datetime import datetime

def log_negative_feedback(user_id, question, answer, file="feedback_log.csv"):
    file_exists = os.path.exists(file)

    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # write header once
        if not file_exists:
            writer.writerow(["timestamp", "user_id", "question", "answer", "feedback"])

        writer.writerow([
            datetime.utcnow().isoformat(),
            user_id,
            question,
            answer,
            "negative"
        ])
