# RAG FastAPI Implementation

This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI, allowing you to search through documents via REST API endpoints.

## Features

- Search documents using semantic similarity
- REST API endpoints for easy integration
- FastAPI automatic API documentation
- Built with sentence transformers for efficient embeddings

## API Endpoints

- `GET /` - Root endpoint with API information
- `POST /search` - Search for relevant documents based on query
- `GET /documents` - Get all documents in the knowledge base
- `GET /health` - Health check endpoint
- `GET /docs` - Auto-generated API documentation

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR
   # venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`

## Usage Example

To search for documents using curl:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "top_k": 3
  }'
```

Or use any HTTP client to send a POST request to `/search` with the JSON payload:
```json
{
  "query": "Your search query here",
  "top_k": 3
}
```

## Project Structure

- `main.py` - FastAPI application with API endpoints
- `rag.py` - Core RAG functionality
- `documents.py` - Sample document collection
- `embeddings.py` - Embedding and similarity computation
- `requirements.txt` - Project dependencies