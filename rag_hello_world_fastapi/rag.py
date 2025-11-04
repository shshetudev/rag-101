"""
RAG (Retrieval-Augmented Generation) core functionality for FastAPI.
This module contains the main RAG class that combines retrieval and generation.
"""
from documents import DOCUMENTS
from embeddings import SimpleEmbedder, find_most_similar_documents


class SimpleRAG:
    def __init__(self):
        """
        Initialize the RAG system with documents and embeddings.
        """
        self.documents = DOCUMENTS
        self.embedder = SimpleEmbedder()
        
        # Pre-compute embeddings for all documents for efficiency
        print("Pre-computing document embeddings...")
        self.document_contents = [doc['content'] for doc in self.documents]
        self.document_embeddings = self.embedder.embed_texts(self.document_contents)
        print(f"Computed embeddings for {len(self.documents)} documents.")
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve the top_k most relevant documents for the given query.
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)
        
        # Find most similar documents
        results = find_most_similar_documents(
            query_embedding, 
            self.document_embeddings, 
            self.documents, 
            top_k=top_k
        )
        
        return results
    
    def generate(self, query, context_docs):
        """
        Generate a response based on the query and retrieved documents.
        This is a simple approach that combines context and generates a response.
        """
        # Create a context string from the retrieved documents
        context_str = "Relevant information:\\n"
        for i, doc in enumerate(context_docs):
            context_str += f"{i+1}. {doc['document']['title']}: {doc['document']['content']}\\n"
        
        # Simple generation - just return the context with the query
        response = f"Query: {query}\\n\\n{context_str}\\n\\nBased on the provided information, this is the most relevant content related to your query."
        return response
    
    def query(self, query, top_k=3):
        """
        Complete RAG pipeline: retrieve relevant documents and generate a response.
        """
        # Step 1: Retrieve relevant documents
        relevant_docs = self.retrieve(query, top_k)
        
        # Step 2: Generate response based on retrieved documents
        response = self.generate(query, relevant_docs)
        
        return {
            'query': query,
            'response': response,
            'relevant_documents': relevant_docs
        }

# Create a global instance of the RAG system to be used by the API
rag_system = SimpleRAG()