"""Embedding generation using sentence transformers and LangChain."""
from typing import List
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class EmbeddingGenerator:
    """Generates embeddings for text using sentence transformers via LangChain."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embeddings.embed_query(text)
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings
    
    def generate_document_embeddings(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for LangChain documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of embedding vectors
        """
        texts = [doc.page_content for doc in documents]
        return self.generate_embeddings_batch(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by the model.
        
        Returns:
            Embedding dimension
        """
        sample_embedding = self.generate_embedding("test")
        return len(sample_embedding)
