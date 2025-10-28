"""
Database Manager Module

Handles all Cosmos DB operations for document metadata storage.
Based on the existing CosmosDBEmbeddingHandler from the notebook.

This module provides CRUD operations for document metadata including:
- Creating new document records
- Retrieving document information
- Updating document metadata
- Deleting document records
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from loguru import logger

from config.settings import settings


class DatabaseManager:
    """
    Manages Cosmos DB operations for document metadata.
    
    This class provides a unified interface for interacting with Cosmos DB,
    based on the existing CosmosDBEmbeddingHandler implementation.
    """
    
    def __init__(self):
        """Initialize Cosmos DB connection and create containers if needed."""
        self.cosmos_endpoint = settings.cosmos_endpoint
        self.cosmos_key = settings.cosmos_key
        self.database_name = settings.cosmos_database
        self.container_name = settings.cosmos_container
        
        # Initialize Cosmos client
        self.client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
        self.database = self.client.create_database_if_not_exists(id=self.database_name)
        self.container = self.database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/id"),
        )
        
        logger.info(f"✅ Connected to Cosmos DB: {self.database_name}/{self.container_name}")
    
    def create_document(
        self,
        filename: str,
        file_path: str,
        file_size: int,
        file_type: str,
        project_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new document record in Cosmos DB.
        
        Args:
            filename: Original filename
            file_path: Path where file is stored
            file_size: Size of file in bytes
            file_type: File extension/type
            project_name: Optional project categorization
            metadata: Additional metadata as dictionary
        
        Returns:
            Dict: Created document record
        
        Raises:
            Exception: If database operation fails
        """
        try:
            document = {
                "id": str(uuid.uuid4()),
                "filename": filename,
                "file_path": file_path,
                "file_size": file_size,
                "file_type": file_type,
                "project_name": project_name,
                "upload_date": datetime.utcnow().isoformat() + "Z",
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "status": "pending",
                "chunk_count": 0,
                "metadata": metadata or {}
            }
            
            created_doc = self.container.create_item(body=document)
            logger.info(f"Created document record: {created_doc['id']} - {filename}")
            return created_doc
        
        except Exception as e:
            logger.error(f"Failed to create document record: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document UUID
        
        Returns:
            Dict: Document record or None if not found
        """
        try:
            document = self.container.read_item(item=doc_id, partition_key=doc_id)
            return document
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Document {doc_id} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None
    
    def get_all_documents(
        self,
        project_name: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all documents with optional filtering.
        
        Args:
            project_name: Filter by project name
            status: Filter by processing status
        
        Returns:
            List[Dict]: List of document records
        """
        try:
            query_parts = []
            parameters = []
            
            if project_name:
                query_parts.append("c.project_name = @project_name")
                parameters.append({"name": "@project_name", "value": project_name})
            
            if status:
                query_parts.append("c.status = @status")
                parameters.append({"name": "@status", "value": status})
            
            where_clause = " AND ".join(query_parts) if query_parts else ""
            query = f"SELECT * FROM c {f'WHERE {where_clause}' if where_clause else ''} ORDER BY c.upload_date DESC"
            
            documents = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            return documents
        
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def update_document(
        self,
        doc_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Update document metadata.
        
        Args:
            doc_id: Document UUID
            **kwargs: Fields to update (embedding_id, status, chunk_count, etc.)
        
        Returns:
            Dict: Updated document record or None if not found
        
        Raises:
            Exception: If database operation fails
        """
        try:
            # Get existing document
            document = self.get_document(doc_id)
            if not document:
                logger.warning(f"Document {doc_id} not found for update")
                return None
            
            # Update fields
            for key, value in kwargs.items():
                if key in document:
                    document[key] = value
            
            document["last_updated"] = datetime.utcnow().isoformat() + "Z"
            
            # Update in Cosmos DB
            updated_doc = self.container.replace_item(item=doc_id, body=document)
            logger.info(f"Updated document {doc_id}")
            return updated_doc
        
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document record from Cosmos DB.
        
        Args:
            doc_id: Document UUID
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            self.container.delete_item(item=doc_id, partition_key=doc_id)
            logger.info(f"Deleted document {doc_id}")
            return True
        
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Document {doc_id} not found for deletion")
            return False
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def update_embedding_status(
        self,
        doc_id: str,
        chunk_count: int,
        status: str = "completed"
    ) -> Optional[Dict[str, Any]]:
        """
        Update document with embedding information after successful processing.
        
        Args:
            doc_id: Document UUID
            chunk_count: Number of chunks created
            status: Processing status (default: "completed")
        
        Returns:
            Dict: Updated document or None if not found
        """
        return self.update_document(
            doc_id=doc_id,
            chunk_count=chunk_count,
            status=status
        )
    
    def get_document_count(self) -> int:
        """
        Get total number of documents in the container.
        
        Returns:
            int: Document count
        """
        try:
            query = "SELECT VALUE COUNT(1) FROM c"
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items[0] if items else 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0


# Global database manager instance
db_manager = DatabaseManager()

