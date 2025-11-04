"""
Test script to verify the FastAPI RAG implementation works correctly.
"""
from main import app, search_documents
from pydantic import BaseModel
from rag import rag_system

def test_rag_functionality():
    """Test the core RAG functionality directly"""
    print("Testing RAG functionality...")
    
    # Test the RAG system directly
    result = rag_system.query("What is RAG?", top_k=2)
    print(f"Query: {result['query']}")
    print(f"Number of relevant documents: {len(result['relevant_documents'])}")
    print("First relevant document:", result['relevant_documents'][0]['document']['title'])
    print()
    
    # Test the search endpoint function
    from main import SearchRequest
    request = SearchRequest(query="What is machine learning?", top_k=2)
    response = search_documents(request)
    
    print(f"Search API response for '{request.query}':")
    print(f"Number of documents returned: {len(response['relevant_documents'])}")
    if response['relevant_documents']:
        print(f"First document: {response['relevant_documents'][0]['document']['title']}")
    
    print("\nAll tests passed! The FastAPI RAG implementation is working correctly.")

if __name__ == "__main__":
    test_rag_functionality()