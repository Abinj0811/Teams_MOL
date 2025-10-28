"""
Delete Manager Module

Handles deletion of documents and their associated embeddings.
Orchestrates the complete cleanup process.

This module provides:
- Document validation and retrieval
- Embedding deletion from Cosmos DB
- File cleanup from local storage
- Metadata deletion
"""

import os
from pathlib import Path
from typing import Dict, Any
from fastapi import HTTPException
from loguru import logger

from modules.db_manager import db_manager
from services.vector_store_service import vector_store_service


class DeleteManager:
    """
    Manages the complete deletion pipeline for documents.
    
    This class orchestrates the process of removing documents
    and their associated embeddings from both Cosmos DB and local storage.
    """
    
    def __init__(self):
        """Initialize delete manager."""
        logger.info("DeleteManager initialized")
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document and all its associated data.
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            Dict: Deletion result with status
        
        Raises:
            HTTPException: If deletion fails
        """
        try:
            logger.info(f"Deleting document: {doc_id}")
            
            # Step 1: Check if document exists
            existing_doc = db_manager.get_document(doc_id)
            if not existing_doc:
                raise HTTPException(status_code=404, detail="Document not found")
            
            filename = existing_doc.get("filename", "Unknown")
            file_path = existing_doc.get("file_path")
            
            try:
                # Step 2: Delete embeddings from Cosmos DB
                embeddings_deleted = vector_store_service.delete_embeddings(doc_id)
                
                # Step 3: Delete document metadata from database
                metadata_deleted = db_manager.delete_document(doc_id)
                
                # Step 4: Clean up local file
                file_deleted = False
                if file_path and Path(file_path).exists():
                    try:
                        os.remove(file_path)
                        file_deleted = True
                        logger.info(f"Deleted local file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete local file {file_path}: {e}")
                
                # Verify deletion success
                if not embeddings_deleted or not metadata_deleted:
                    logger.error(f"Partial deletion failure for document {doc_id}")
                    raise HTTPException(status_code=500, detail="Partial deletion failure")
                
                logger.info(f"Successfully deleted document {doc_id}")
                
                return {
                    "doc_id": doc_id,
                    "filename": filename,
                    "status": "deleted",
                    "embeddings_deleted": embeddings_deleted,
                    "metadata_deleted": metadata_deleted,
                    "file_deleted": file_deleted
                }
            
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete document {doc_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete failed for document {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """
        Get document information before deletion.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Dict: Document information
        
        Raises:
            HTTPException: If document not found
        """
        try:
            document = db_manager.get_document(doc_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            return {
                "doc_id": doc_id,
                "filename": document.get("filename"),
                "file_size": document.get("file_size"),
                "file_type": document.get("file_type"),
                "project_name": document.get("project_name"),
                "upload_date": document.get("upload_date"),
                "chunk_count": document.get("chunk_count"),
                "status": document.get("status")
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document info for {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")


# Global delete manager instance
delete_manager = DeleteManager()

