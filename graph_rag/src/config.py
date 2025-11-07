"""Configuration management for the Graph RAG application."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Neo4j Configuration
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password123")
    
    # Embedding Model Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Text Processing Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # API Configuration
    api_title: str = "Graph RAG API"
    api_version: str = "1.0.0"
    api_description: str = "A Graph-based Retrieval Augmented Generation system using Neo4j and LangChain"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
