"""
FastAPI application for the RAG (Retrieval-Augmented Generation) system.
Provides REST API endpoints for document search and retrieval.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from rag import rag_system

# Create FastAPI instance
app = FastAPI(
    title="RAG API",
    description="A simple Retrieval-Augmented Generation API that allows searching through documents",
    version="1.0.0"
)

# Define Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class Document(BaseModel):
    id: int
    title: str
    content: str

class RetrievedDocument(BaseModel):
    document: Document
    similarity: float

class SearchResponse(BaseModel):
    query: str
    response: str
    relevant_documents: List[RetrievedDocument]

# Define API endpoints
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the RAG API",
        "description": "This API allows you to search through documents using RAG technology",
        "endpoints": {
            "/search": "POST endpoint to search documents",
            "/docs": "Auto-generated API documentation"
        }
    }

@app.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):
    """
    Search for relevant documents based on the query.
    
    - **query**: The search query text
    - **top_k**: Number of top documents to retrieve (default: 3)
    """
    # Use the RAG system to process the query
    result = rag_system.query(request.query, request.top_k)
    return result

@app.get("/documents")
def get_documents():
    """
    Get all documents in the knowledge base.
    """
    return {"documents": rag_system.documents}

@app.get("/health")
def health_check():
    """
    Check if the API is running.
    """
    return {"status": "healthy", "message": "RAG API is running"}