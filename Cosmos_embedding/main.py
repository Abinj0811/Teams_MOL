"""
FastAPI Main Application

Cosmos Embedding Admin Panel - FastAPI entry point with all endpoints.
Based on the existing notebook workflow converted to production-ready API.

Endpoints:
- POST /upload - Upload new documents and create embeddings
- PUT /update/{doc_id} - Update existing documents and re-embed
- DELETE /delete/{doc_id} - Delete documents and embeddings
- GET /health - Health check endpoint
- GET /documents - List all documents
- GET /documents/{doc_id} - Get specific document info
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from loguru import logger
import os

# Import managers and services
from modules.upload_manager import upload_manager
from modules.update_manager import update_manager
from modules.delete_manager import delete_manager
from modules.db_manager import db_manager
from services.vector_store_service import vector_store_service
from config.settings import settings


# Pydantic models for request/response validation
class UploadResponse(BaseModel):
    """Response model for document upload."""
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    file_size: int
    project_name: Optional[str] = None


class UpdateResponse(BaseModel):
    """Response model for document update."""
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    file_size: int


class DeleteResponse(BaseModel):
    """Response model for document deletion."""
    doc_id: str
    filename: str
    status: str
    embeddings_deleted: bool
    metadata_deleted: bool
    file_deleted: bool


class DocumentInfo(BaseModel):
    """Document information model."""
    doc_id: str
    filename: str
    file_size: int
    file_type: str
    project_name: Optional[str] = None
    upload_date: str
    chunk_count: int
    status: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    app_name: str
    version: str
    cosmos_connected: bool
    document_count: int
    embedding_count: int


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Admin Panel for managing document embeddings in Cosmos DB",
    debug=settings.debug_mode
)

# Optional API key authentication
def verify_api_key(api_key: Optional[str] = Query(None)):
    """Optional API key verification."""
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify system status.
    
    Returns:
        HealthResponse: System health information
    """
    try:
        # Check Cosmos DB connection
        cosmos_connected = True
        try:
            document_count = db_manager.get_document_count()
            embedding_count = vector_store_service.get_document_count()
        except Exception as e:
            logger.warning(f"Cosmos DB connection issue: {e}")
            cosmos_connected = False
            document_count = 0
            embedding_count = 0
        
        return HealthResponse(
            status="healthy" if cosmos_connected else "degraded",
            app_name=settings.app_name,
            version=settings.app_version,
            cosmos_connected=cosmos_connected,
            document_count=document_count,
            embedding_count=embedding_count
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            app_name=settings.app_name,
            version=settings.app_version,
            cosmos_connected=False,
            document_count=0,
            embedding_count=0
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    project_name: Optional[str] = Query(None, description="Optional project categorization"),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Upload a new document and create embeddings.
    
    Args:
        file: Document file (PDF, DOCX, TXT, MD)
        project_name: Optional project categorization
        api_key: Optional API key for authentication
    
    Returns:
        UploadResponse: Upload result with document ID and status
    """
    try:
        logger.info(f"Upload request received: {file.filename}")
        
        result = upload_manager.upload_document(
            file=file,
            project_name=project_name
        )
        
        return UploadResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.put("/update/{doc_id}", response_model=UpdateResponse)
async def update_document(
    doc_id: str,
    file: UploadFile = File(...),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Update an existing document with new content.
    
    Args:
        doc_id: Document ID to update
        file: New document file content
        api_key: Optional API key for authentication
    
    Returns:
        UpdateResponse: Update result with status
    """
    try:
        logger.info(f"Update request received for doc {doc_id}: {file.filename}")
        
        result = update_manager.update_document(
            doc_id=doc_id,
            file=file
        )
        
        return UpdateResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.delete("/delete/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Delete a document and all its associated embeddings.
    
    Args:
        doc_id: Document ID to delete
        api_key: Optional API key for authentication
    
    Returns:
        DeleteResponse: Deletion result with status
    """
    try:
        logger.info(f"Delete request received for doc {doc_id}")
        
        result = delete_manager.delete_document(doc_id=doc_id)
        
        return DeleteResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    project_name: Optional[str] = Query(None, description="Filter by project name"),
    status: Optional[str] = Query(None, description="Filter by status"),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    List all documents with optional filtering.
    
    Args:
        project_name: Optional project filter
        status: Optional status filter
        api_key: Optional API key for authentication
    
    Returns:
        List[DocumentInfo]: List of documents
    """
    try:
        documents = db_manager.get_all_documents(
            project_name=project_name,
            status=status
        )
        
        return [DocumentInfo(**doc) for doc in documents]
    
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Get information about a specific document.
    
    Args:
        doc_id: Document ID
        api_key: Optional API key for authentication
    
    Returns:
        DocumentInfo: Document information
    """
    try:
        document = db_manager.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentInfo(**document)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@app.get("/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return"),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Search documents using text-based search.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        api_key: Optional API key for authentication
    
    Returns:
        List: Search results
    """
    try:
        results = vector_store_service.search_documents(query=query, top_k=top_k)
        return {"query": query, "results": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug_mode}")
    logger.info(f"Upload folder: {settings.upload_folder}")
    logger.info(f"Cosmos DB: {settings.cosmos_database}/{settings.cosmos_container}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info(f"Shutting down {settings.app_name}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug_mode,
        log_level="info"
    )
