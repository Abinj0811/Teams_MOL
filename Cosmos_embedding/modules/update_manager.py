"""
Update Manager Module

Handles updating existing documents and re-generating embeddings.
Orchestrates the complete pipeline for document updates.

This module provides:
- Document validation and retrieval
- File replacement and processing
- Embedding re-generation
- Cosmos DB update operations
- Metadata updates
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


class UpdateManager:
    """
    Manages the complete update pipeline for existing documents.
    
    This class orchestrates the process of updating an existing document
    and re-generating its embeddings in Cosmos DB.
    """
    
    def __init__(self):
        """Initialize update manager."""
        self.upload_folder = Path(settings.upload_folder)
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        logger.info("UpdateManager initialized")
    
    def validate_file(self, file: UploadFile) -> None:
        """
        Validate uploaded file for update.
        
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
        Save updated file to local storage.
        
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
            
            logger.info(f"Saved updated file: {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Failed to save updated file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save updated file")
    
    def update_document(
        self,
        doc_id: str,
        file: UploadFile
    ) -> Dict[str, Any]:
        """
        Update an existing document with new content.
        
        Args:
            doc_id: Document ID to update
            file: New file content
        
        Returns:
            Dict: Update result with status
        
        Raises:
            HTTPException: If update fails
        """
        try:
            logger.info(f"Updating document: {doc_id}")
            
            # Step 1: Validate file
            self.validate_file(file)
            
            # Step 2: Check if document exists
            existing_doc = db_manager.get_document(doc_id)
            if not existing_doc:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Step 3: Save new file
            new_file_path = self.save_file(file)
            
            try:
                # Step 4: Update document status to processing
                db_manager.update_document(
                    doc_id=doc_id,
                    status="updating",
                    file_path=new_file_path,
                    filename=file.filename
                )
                
                # Step 5: Process new document content
                chunks, embeddings = embedding_utils.process_document(new_file_path)
                
                # Step 6: Update embeddings in Cosmos DB
                vector_store_service.update_embeddings(
                    doc_id=doc_id,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata={
                        "filename": file.filename,
                        "project_name": existing_doc.get("project_name"),
                        "file_type": Path(file.filename).suffix.lower().lstrip('.'),
                        "chunk_count": len(chunks),
                        "updated": True
                    }
                )
                
                # Step 7: Update document metadata
                db_manager.update_embedding_status(
                    doc_id=doc_id,
                    chunk_count=len(chunks),
                    status="completed"
                )
                
                # Step 8: Clean up old file if different path
                old_file_path = existing_doc.get("file_path")
                if old_file_path and old_file_path != new_file_path:
                    try:
                        if Path(old_file_path).exists():
                            os.remove(old_file_path)
                            logger.info(f"Cleaned up old file: {old_file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up old file {old_file_path}: {e}")
                
                logger.info(f"Successfully updated document {doc_id}: {len(chunks)} chunks")
                
                return {
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "status": "updated",
                    "chunk_count": len(chunks),
                    "file_size": Path(new_file_path).stat().st_size
                }
            
            except Exception as e:
                # Update status to failed
                db_manager.update_document(doc_id, status="failed")
                logger.error(f"Failed to update document {doc_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Update failed for document {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


# Global update manager instance
update_manager = UpdateManager()

