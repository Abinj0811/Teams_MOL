"""
Upload Manager Module

Handles file upload and embedding creation for new documents.
Orchestrates the complete pipeline from file upload to Cosmos DB storage.

This module provides:
- File validation and storage
- Text extraction and chunking
- Embedding generation
- Cosmos DB storage
- Metadata management
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import UploadFile, HTTPException
from loguru import logger

from modules.embedding_utils import embedding_utils
from modules.db_manager import db_manager
from services.vector_store_service import vector_store_service
from config.settings import settings


class UploadManager:
    """
    Manages the complete upload pipeline for new documents.
    
    This class orchestrates the entire process from file upload to
    embedding storage in Cosmos DB, based on the notebook workflow.
    """
    
    def __init__(self):
        """Initialize upload manager."""
        self.upload_folder = Path(settings.upload_folder)
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        logger.info("UploadManager initialized")
    
    def validate_file(self, file: UploadFile) -> None:
        """
        Validate uploaded file.
        
        Args:
            file: Uploaded file object
        
        Raises:
            HTTPException: If file validation fails
        """
        # Check file size
        if file.size and file.size > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Check file extension
        if file.filename:
            file_ext = Path(file.filename).suffix.lower().lstrip('.')
            if file_ext not in settings.allowed_extensions_list:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed: {settings.allowed_extensions}"
                )
    
    def save_file(self, file: UploadFile) -> str:
        """
        Save uploaded file to local storage.
        
        Args:
            file: Uploaded file object
        
        Returns:
            str: Path to saved file
        
        Raises:
            HTTPException: If file saving fails
        """
        try:
            # Generate unique filename
            file_ext = Path(file.filename).suffix
            unique_filename = f"{file.filename}_{os.urandom(8).hex()}{file_ext}"
            file_path = self.upload_folder / unique_filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"Saved file: {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save file")
    
    def process_document(
        self,
        file_path: str,
        filename: str,
        project_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process document through the complete pipeline.
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            project_name: Optional project categorization
        
        Returns:
            Dict: Processing result with document ID and status
        
        Raises:
            HTTPException: If processing fails
        """
        try:
            logger.info(f"Processing document: {filename}")
            
            # Get file info
            file_size = Path(file_path).stat().st_size
            file_type = Path(file_path).suffix.lower().lstrip('.')
            
            # Step 1: Create document record in database
            document = db_manager.create_document(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                project_name=project_name,
                metadata={"processing_status": "started"}
            )
            
            doc_id = document["id"]
            
            try:
                # Step 2: Extract text and generate embeddings
                chunks, embeddings = embedding_utils.process_document(file_path)
                
                # Step 3: Store embeddings in Cosmos DB
                vector_store_service.add_embeddings(
                    doc_id=doc_id,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata={
                        "filename": filename,
                        "project_name": project_name,
                        "file_type": file_type,
                        "chunk_count": len(chunks)
                    }
                )
                
                # Step 4: Update document status
                db_manager.update_embedding_status(
                    doc_id=doc_id,
                    chunk_count=len(chunks),
                    status="completed"
                )
                
                logger.info(f"Successfully processed document {doc_id}: {len(chunks)} chunks")
                
                return {
                    "doc_id": doc_id,
                    "filename": filename,
                    "status": "completed",
                    "chunk_count": len(chunks),
                    "file_size": file_size,
                    "project_name": project_name
                }
            
            except Exception as e:
                # Update status to failed
                db_manager.update_document(doc_id, status="failed")
                logger.error(f"Failed to process document {doc_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Failed to process document {filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")
    
    def upload_document(
        self,
        file: UploadFile,
        project_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete upload pipeline for a new document.
        
        Args:
            file: Uploaded file object
            project_name: Optional project categorization
        
        Returns:
            Dict: Upload result with document ID and status
        
        Raises:
            HTTPException: If upload fails
        """
        try:
            # Step 1: Validate file
            self.validate_file(file)
            
            # Step 2: Save file
            file_path = self.save_file(file)
            
            # Step 3: Process document
            result = self.process_document(
                file_path=file_path,
                filename=file.filename,
                project_name=project_name
            )
            
            logger.info(f"Upload completed successfully: {result['doc_id']}")
            return result
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Global upload manager instance
upload_manager = UploadManager()

