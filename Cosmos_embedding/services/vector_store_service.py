"""
Vector Store Service Module

Handles all operations related to Cosmos DB vector storage and retrieval.
Based on the existing CosmosDBEmbeddingHandler from the notebook.

This service provides:
- Adding document embeddings to Cosmos DB
- Updating existing embeddings
- Deleting embeddings
- Querying/searching for similar documents
"""

import uuid
from typing import List, Dict, Optional, Any
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from loguru import logger

from config.settings import settings


class VectorStoreService:
    """
    Service for managing Cosmos DB vector store operations.
    
    This class abstracts the Cosmos DB vector store implementation,
    based on the existing CosmosDBEmbeddingHandler from the notebook.
    """
    
    def __init__(self):
        """Initialize Cosmos DB vector store."""
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
        
        logger.info(f"VectorStoreService initialized with Cosmos DB: {self.database_name}/{self.container_name}")
    
    def add_embeddings(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add document embeddings to Cosmos DB.
        Based on the create_cosmos_vectorstore method from the notebook.
        
        Args:
            doc_id: Unique document identifier
            chunks: List of text chunks
            embeddings: List of embedding vectors corresponding to chunks
            metadata: Optional metadata to store with embeddings
        
        Returns:
            str: Container name for reference
        
        Raises:
            Exception: If adding embeddings fails
        """
        try:
            logger.info(f"Adding {len(chunks)} embeddings to Cosmos DB for doc {doc_id}")
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,  # Reference to parent document
                    "chunk_index": i,
                    "text": chunk,
                    "vector_embedding": embedding,  # Vector field for Cosmos DB vector index
                    "metadata": metadata or {}
                }
                
                self.container.upsert_item(doc)
            
            logger.info(f"✅ Uploaded {len(chunks)} vectors to Cosmos DB container '{self.container_name}'")
            return self.container_name
        
        except Exception as e:
            logger.error(f"Failed to add embeddings for doc {doc_id}: {e}")
            raise
    
    def update_embeddings(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update existing document embeddings.
        
        Strategy: Delete old embeddings and add new ones.
        
        Args:
            doc_id: Document ID
            chunks: New text chunks
            embeddings: New embedding vectors
            metadata: Updated metadata
        
        Returns:
            bool: True if successful
        """
        try:
            # Delete old embeddings
            self.delete_embeddings(doc_id)
            
            # Add new embeddings
            self.add_embeddings(doc_id, chunks, embeddings, metadata)
            
            logger.info(f"Updated embeddings for doc {doc_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update embeddings for doc {doc_id}: {e}")
            return False
    
    def delete_embeddings(self, doc_id: str) -> bool:
        """
        Delete all embeddings associated with a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            bool: True if successful
        """
        try:
            # Query all chunks for this document
            query = "SELECT c.id FROM c WHERE c.doc_id = @doc_id"
            parameters = [{"name": "@doc_id", "value": doc_id}]
            
            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if results:
                # Delete all matching chunks
                for result in results:
                    chunk_id = result["id"]
                    self.container.delete_item(item=chunk_id, partition_key=chunk_id)
                
                logger.info(f"Deleted {len(results)} chunks from Cosmos DB for doc {doc_id}")
            else:
                logger.warning(f"No embeddings found for doc {doc_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete embeddings for doc {doc_id}: {e}")
            return False
    
    def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using text-based search.
        Based on the search_documents method from the notebook.
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            List of matching documents with metadata and scores
        """
        try:
            # Extract key terms from the query for text search
            search_terms = query.lower().split()
            search_term = " ".join(search_terms[:3])  # Use first 3 terms
            
            search_query = f"""
                SELECT TOP {top_k} c.id, c.text, c.doc_id, c.chunk_index, c.metadata
                FROM c
                WHERE CONTAINS(LOWER(c.text), LOWER(@search_term))
            """
            
            parameters = [{"name": "@search_term", "value": search_term}]
            
            results = list(self.container.query_items(
                query=search_query,
                parameters=parameters,
                enable_cross_partition_query=True,
            ))
            
            # If no results from text search, return all documents (fallback)
            if not results:
                fallback_query = f"SELECT TOP {top_k} c.id, c.text, c.doc_id, c.chunk_index, c.metadata FROM c"
                results = list(self.container.query_items(
                    query=fallback_query,
                    enable_cross_partition_query=True,
                ))
            
            # Format results to include score for compatibility
            formatted_results = []
            for i, result in enumerate(results):
                formatted_result = {
                    "id": result.get("id"),
                    "text": result.get("text", ""),
                    "doc_id": result.get("doc_id"),
                    "chunk_index": result.get("chunk_index"),
                    "metadata": result.get("metadata", {}),
                    "score": 1.0 - (i * 0.1),  # Simple scoring based on order
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"🔍 Found {len(formatted_results)} similar documents")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def get_document_count(self) -> int:
        """
        Get total number of embedding chunks in the container.
        
        Returns:
            int: Embedding count
        """
        try:
            query = "SELECT VALUE COUNT(1) FROM c"
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items[0] if items else 0
        except Exception as e:
            logger.error(f"Failed to get embedding count: {e}")
            return 0


# Global vector store service instance
vector_store_service = VectorStoreService()

