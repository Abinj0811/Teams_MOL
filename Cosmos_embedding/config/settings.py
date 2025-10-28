"""
Configuration settings for the Cosmos Embedding Admin Panel.
Loads environment variables and provides centralized configuration management.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Azure Cosmos DB Configuration
    cosmos_endpoint: str
    cosmos_key: str
    cosmos_database: str = "EmbeddingsDB"
    cosmos_container: str = "ragEmbedding"
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "text-embedding-3-large"
    
    # File Upload Configuration
    max_file_size_mb: int = 50
    allowed_extensions: str = "pdf,docx,txt,md"
    upload_folder: str = "./static/uploads"
    
    # Application Configuration
    app_name: str = "Cosmos Embedding Admin Panel"
    app_version: str = "1.0.0"
    debug_mode: bool = True
    
    # Optional API Authentication
    api_key: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Get list of allowed file extensions."""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)


# Create global settings instance
settings = Settings()

# Ensure required directories exist
settings.ensure_directories()

