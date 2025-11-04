"""
Embedding functionality for our RAG FastAPI system.
This uses sentence transformers to create embeddings and cosine similarity to find relevant documents.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedder with a pre-trained sentence transformer model.
        The 'all-MiniLM-L6-v2' model is lightweight but effective for demonstration purposes.
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text):
        """
        Create an embedding vector for the given text.
        """
        return self.model.encode([text])[0]  # Return first (and only) embedding as 1D array
    
    def embed_texts(self, texts):
        """
        Create embeddings for multiple texts at once.
        """
        return self.model.encode(texts)
    
    def cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two embedding vectors.
        Returns a value between -1 and 1, where 1 means identical.
        """
        # Reshape for sklearn (expecting 2D arrays)
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

def find_most_similar_documents(query_embedding, document_embeddings, documents, top_k=3):
    """
    Find the top_k most similar documents to the query based on embeddings.
    """
    similarities = []
    for doc_emb in document_embeddings:
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1), 
            doc_emb.reshape(1, -1)
        )[0][0]
        similarities.append(similarity)
    
    # Get indices of top_k most similar documents
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return the documents with their similarity scores
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx]
        })
    
    return results