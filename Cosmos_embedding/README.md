# Cosmos Embedding Admin Panel

A production-ready FastAPI-based admin panel for managing document embeddings in Azure Cosmos DB. This system converts the existing Jupyter notebook workflow into a clean, modular API that supports uploading, updating, and deleting documents with automatic embedding generation.

## 🚀 Features

- **Document Upload**: Upload PDF, DOCX, TXT, and MD files with automatic text extraction and embedding generation
- **Document Update**: Update existing documents with new content and re-generate embeddings
- **Document Deletion**: Remove documents and their associated embeddings from Cosmos DB
- **Text Extraction**: Advanced text extraction using Docling with support for complex PDF layouts
- **Embedding Generation**: OpenAI embeddings using `text-embedding-3-large` model
- **Cosmos DB Integration**: Native Azure Cosmos DB integration for vector storage
- **Health Monitoring**: Built-in health checks and system status monitoring
- **API Authentication**: Optional API key-based authentication
- **Comprehensive Logging**: Detailed logging with Loguru for production monitoring

## 📁 Project Structure

```
Cosmos_embedding/
│
├── main.py                          # FastAPI entry point with all endpoints
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Configuration management with Pydantic
│
├── modules/
│   ├── __init__.py
│   ├── upload_manager.py            # Handles file upload + embedding creation
│   ├── update_manager.py            # Handles updating existing documents
│   ├── delete_manager.py            # Handles deletion from Cosmos DB
│   ├── embedding_utils.py           # Text extraction + embedding generation
│   └── db_manager.py                # Cosmos DB connection and CRUD operations
│
├── services/
│   ├── __init__.py
│   └── vector_store_service.py      # Handles Cosmos DB vector operations
│
├── static/uploads/                  # Folder for uploaded documents
│
├── .env                             # Environment variables (credentials)
├── .env.example                     # Environment variables template
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🛠️ Installation

### 1. Clone and Setup

```bash
cd Cosmos_embedding
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```env
# Azure Cosmos DB Configuration
COSMOS_ENDPOINT=https://your-account.documents.azure.com:443/
COSMOS_KEY=your-cosmos-primary-key-here
COSMOS_DATABASE=EmbeddingsDB
COSMOS_CONTAINER=ragEmbedding

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=text-embedding-3-large

# File Upload Configuration
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=pdf,docx,txt,md
UPLOAD_FOLDER=./static/uploads

# Application Configuration
APP_NAME=Cosmos Embedding Admin Panel
APP_VERSION=1.0.0
DEBUG_MODE=True

# Optional API Authentication
API_KEY=your-admin-api-key-here
```

### 3. Run the Application

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Endpoints

### Health Check
- **GET** `/health` - System health and status information

### Document Management
- **POST** `/upload` - Upload new documents and create embeddings
- **PUT** `/update/{doc_id}` - Update existing documents and re-embed
- **DELETE** `/delete/{doc_id}` - Delete documents and embeddings
- **GET** `/documents` - List all documents (with optional filtering)
- **GET** `/documents/{doc_id}` - Get specific document information

### Search
- **GET** `/search` - Search documents using text-based queries

### API Documentation
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation

## 🔧 Usage Examples

### Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "project_name=legal_docs"
```

### Update a Document

```bash
curl -X PUT "http://localhost:8000/update/{doc_id}" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@updated_document.pdf"
```

### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/delete/{doc_id}"
```

### List Documents

```bash
curl -X GET "http://localhost:8000/documents?project_name=legal_docs"
```

### Search Documents

```bash
curl -X GET "http://localhost:8000/search?query=authority%20regulations&top_k=5"
```

## 🔐 Authentication

The API supports optional API key authentication. If configured in `.env`, include the API key as a query parameter:

```bash
curl -X GET "http://localhost:8000/documents?api_key=your-api-key"
```

## 🏗️ Architecture

### Core Components

1. **FastAPI Application** (`main.py`)
   - RESTful API endpoints
   - Request/response validation with Pydantic
   - Error handling and logging

2. **Configuration Management** (`config/`)
   - Environment variable loading
   - Settings validation with Pydantic
   - Centralized configuration

3. **Document Processing** (`modules/`)
   - **UploadManager**: Complete upload pipeline
   - **UpdateManager**: Document update workflow
   - **DeleteManager**: Cleanup operations
   - **EmbeddingUtils**: Text extraction and embedding generation
   - **DatabaseManager**: Cosmos DB metadata operations

4. **Vector Storage** (`services/`)
   - **VectorStoreService**: Cosmos DB vector operations
   - Embedding storage and retrieval
   - Document search functionality

### Data Flow

1. **Upload Process**:
   ```
   File Upload → Validation → Local Storage → Text Extraction → 
   Chunking → Embedding Generation → Cosmos DB Storage → Metadata Update
   ```

2. **Update Process**:
   ```
   File Upload → Validation → Delete Old Embeddings → Text Extraction → 
   Chunking → Embedding Generation → Cosmos DB Storage → Metadata Update
   ```

3. **Delete Process**:
   ```
   Document ID → Delete Embeddings → Delete Metadata → Cleanup Local Files
   ```

## 🔍 Text Extraction

The system uses the existing `TextExtraction` class from the notebook, which provides:

- **Docling Integration**: Advanced PDF processing with layout understanding
- **Table Extraction**: Specialized handling for authority and committee tables
- **Multi-format Support**: PDF, DOCX, TXT, MD files
- **Fallback Processing**: Multiple extraction methods for reliability

## 🗄️ Cosmos DB Schema

### Document Metadata Container
```json
{
  "id": "uuid",
  "filename": "document.pdf",
  "file_path": "./static/uploads/document.pdf",
  "file_size": 1024000,
  "file_type": "pdf",
  "project_name": "legal_docs",
  "upload_date": "2024-01-01T00:00:00Z",
  "last_updated": "2024-01-01T00:00:00Z",
  "status": "completed",
  "chunk_count": 15,
  "metadata": {}
}
```

### Embeddings Container
```json
{
  "id": "uuid",
  "doc_id": "parent-document-uuid",
  "chunk_index": 0,
  "text": "Document chunk content...",
  "vector_embedding": [0.1, 0.2, ...],
  "metadata": {
    "filename": "document.pdf",
    "project_name": "legal_docs",
    "file_type": "pdf",
    "chunk_count": 15
  }
}
```

## 🚨 Error Handling

The system includes comprehensive error handling:

- **File Validation**: Size limits, format checking
- **Database Errors**: Connection issues, query failures
- **Processing Errors**: Text extraction failures, embedding generation issues
- **HTTP Exceptions**: Proper status codes and error messages
- **Logging**: Detailed error logging for debugging

## 📊 Monitoring

### Health Check Endpoint

The `/health` endpoint provides:

- Application status
- Cosmos DB connection status
- Document and embedding counts
- System version information

### Logging

The system uses Loguru for structured logging:

- Request/response logging
- Error tracking
- Performance monitoring
- Debug information

## 🔧 Configuration Options

### File Upload Settings
- `MAX_FILE_SIZE_MB`: Maximum file size (default: 50MB)
- `ALLOWED_EXTENSIONS`: Supported file types
- `UPLOAD_FOLDER`: Local storage directory

### Cosmos DB Settings
- `COSMOS_ENDPOINT`: Azure Cosmos DB endpoint
- `COSMOS_KEY`: Access key
- `COSMOS_DATABASE`: Database name
- `COSMOS_CONTAINER`: Container name

### OpenAI Settings
- `OPENAI_API_KEY`: API key for embeddings
- `OPENAI_MODEL`: Embedding model (default: text-embedding-3-large)

## 🚀 Production Deployment

### Environment Variables
Ensure all required environment variables are set in production:

```bash
export COSMOS_ENDPOINT="https://your-account.documents.azure.com:443/"
export COSMOS_KEY="your-production-key"
export OPENAI_API_KEY="your-production-key"
export DEBUG_MODE="False"
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Health Monitoring
Monitor the `/health` endpoint for:
- Application availability
- Cosmos DB connectivity
- System resource usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the logs for error details
2. Verify environment configuration
3. Test Cosmos DB connectivity
4. Check OpenAI API key validity

## 🔄 Migration from Notebook

This FastAPI application is a direct conversion of the existing Jupyter notebook workflow:

- **TextExtraction**: Reused from `chains/text_processor_chain.py`
- **CosmosDBEmbeddingHandler**: Converted to `VectorStoreService`
- **Document Processing Pipeline**: Implemented in manager classes
- **Configuration**: Centralized with Pydantic settings

The system maintains compatibility with existing Cosmos DB data while providing a production-ready API interface.
