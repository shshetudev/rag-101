"""FastAPI application for Graph RAG."""
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import tempfile
from src.config import settings
from src.graph_rag_pipeline import GraphRAGPipeline


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description
)

# Initialize pipeline (will be created per request to avoid connection issues)
def get_pipeline() -> GraphRAGPipeline:
    """Create a new pipeline instance."""
    return GraphRAGPipeline()


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for querying the knowledge graph."""
    query: str = Field(..., description="Query text", min_length=1)
    k: int = Field(5, description="Number of results to return", ge=1, le=20)


class QueryResponse(BaseModel):
    """Response model for query results."""
    query: str
    similar_chunks: list
    query_entities: list
    entity_subgraphs: list


class ProcessResponse(BaseModel):
    """Response model for file processing."""
    file_path: str
    chunks_processed: int
    entities_extracted: int
    relations_extracted: int
    embedding_dimension: int
    status: str


class StatsResponse(BaseModel):
    """Response model for statistics."""
    entities: int
    chunks: int
    relationships: int


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Welcome to Graph RAG API",
        "version": settings.api_version,
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_statistics()
        pipeline.close()
        return {
            "status": "healthy",
            "database": "connected",
            "entities": str(stats["entities"]),
            "chunks": str(stats["chunks"])
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/process/file", response_model=ProcessResponse, tags=["Processing"])
async def process_file(file: UploadFile = File(...)) -> ProcessResponse:
    """
    Process a text file through the Graph RAG pipeline.
    
    - Cleans the text
    - Splits into chunks
    - Extracts entities and relations
    - Generates embeddings
    - Stores in Neo4j
    """
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt files are supported"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process the file
        pipeline = get_pipeline()
        result = pipeline.process_text_file(tmp_file_path)
        pipeline.close()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return ProcessResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/process/text", response_model=ProcessResponse, tags=["Processing"])
async def process_text(text: str) -> ProcessResponse:
    """
    Process raw text through the Graph RAG pipeline.
    
    - Cleans the text
    - Splits into chunks
    - Extracts entities and relations
    - Generates embeddings
    - Stores in Neo4j
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty"
        )
    
    try:
        # Save text to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(text)
            tmp_file_path = tmp_file.name
        
        # Process the file
        pipeline = get_pipeline()
        result = pipeline.process_text_file(tmp_file_path)
        pipeline.close()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return ProcessResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_graph(request: QueryRequest) -> QueryResponse:
    """
    Query the knowledge graph using semantic search and entity subgraphs.
    
    - Performs vector similarity search
    - Extracts entities from query
    - Returns relevant subgraphs
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.query(request.query, k=request.k)
        pipeline.close()
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying graph: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics() -> StatsResponse:
    """
    Get knowledge graph statistics.
    
    Returns counts of entities, chunks, and relationships.
    """
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_statistics()
        pipeline.close()
        
        return StatsResponse(**stats)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving statistics: {str(e)}"
        )


@app.delete("/clear", tags=["Management"])
async def clear_graph() -> Dict[str, str]:
    """
    Clear all data from the knowledge graph.
    
    ⚠️ Warning: This will delete all nodes and relationships!
    """
    try:
        pipeline = get_pipeline()
        pipeline.clear_graph()
        pipeline.close()
        
        return {"message": "Knowledge graph cleared successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing graph: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
