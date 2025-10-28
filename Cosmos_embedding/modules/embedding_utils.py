"""
Embedding Utils Module

Handles text extraction from various document formats and embedding generation.
Based on the existing TextExtraction and CosmosDBEmbeddingHandler from the notebook.

This module provides utilities for:
- Extracting text from PDF, DOCX, TXT, and MD files using Docling
- Chunking text into manageable segments
- Generating embeddings using OpenAI
- Managing embedding operations for document processing
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LCDocument
from loguru import logger

from .text_extraction import TextExtraction

from config.settings import settings


class EmbeddingUtils:
    """
    Utility class for document text extraction and embedding generation.
    
    This class handles the entire pipeline from document to embeddings:
    1. Extract text from various document formats using Docling
    2. Split text into chunks
    3. Generate embeddings for each chunk
    """
    
    def __init__(self):
        """Initialize embedding model and text splitter."""
        self.embeddings_model = self._initialize_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.text_extractor = TextExtraction()
        logger.info("EmbeddingUtils initialized")
    
    def _initialize_embeddings(self):
        """
        Initialize the OpenAI embeddings model.
        
        Returns:
            OpenAIEmbeddings: Embeddings model
        
        Raises:
            ValueError: If no valid API credentials are provided
        """
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not found in configuration")
            
            logger.info("Initializing OpenAI embeddings")
            return OpenAIEmbeddings(
                model=settings.openai_model,
                openai_api_key=settings.openai_api_key
            )
        
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text content from a file based on its extension.
        Uses the existing TextExtraction class from the notebook.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            str: Extracted text content
        
        Raises:
            ValueError: If file type is not supported
            Exception: If extraction fails
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension not in settings.allowed_extensions_list:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Use the existing TextExtraction class from the notebook
            text = self.text_extractor.extract_to_markdown(file_path)
            
            if not text or len(text.strip()) == 0:
                raise ValueError("No text content extracted from document")
            
            logger.info(f"Extracted {len(text)} characters from {file_path}")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks suitable for embedding.
        
        Args:
            text: Input text to chunk
        
        Returns:
            List[str]: List of text chunks
        """
        try:
            # Convert to LangChain Document format
            document = LCDocument(page_content=text)
            chunks = self.text_splitter.split_documents([document])
            
            # Extract text content from chunks
            chunk_texts = [chunk.page_content for chunk in chunks]
            logger.info(f"Split text into {len(chunk_texts)} chunks")
            return chunk_texts
        
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List[List[float]]: List of embedding vectors
        
        Raises:
            Exception: If embedding generation fails
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} text chunks")
            embeddings = self.embeddings_model.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def process_document(self, file_path: str) -> Tuple[List[str], List[List[float]]]:
        """
        Complete pipeline: extract text, chunk it, and generate embeddings.
        
        This is the main method that orchestrates the entire process:
        1. Extract text from document using Docling
        2. Split into chunks
        3. Generate embeddings
        
        Args:
            file_path: Path to document file
        
        Returns:
            Tuple containing:
                - List of text chunks
                - List of embedding vectors
        
        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Step 1: Extract text using existing TextExtraction
            text = self.extract_text_from_file(file_path)
            
            # Step 2: Chunk text
            chunks = self.chunk_text(text)
            
            if not chunks:
                raise ValueError("No chunks created from text")
            
            # Step 3: Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            logger.info(f"Successfully processed document: {len(chunks)} chunks, {len(embeddings)} embeddings")
            return chunks, embeddings
        
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.
        
        Used for query embedding when searching the vector store.
        
        Args:
            query: Query text to embed
        
        Returns:
            List[float]: Embedding vector
        """
        try:
            embedding = self.embeddings_model.embed_query(query)
            logger.debug(f"Generated query embedding for: {query[:50]}...")
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise


# Global embedding utils instance
embedding_utils = EmbeddingUtils()

