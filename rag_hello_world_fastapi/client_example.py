"""
Client script to demonstrate how to interact with the RAG FastAPI application.
This shows how to send requests to the API endpoints.
"""
import requests
import json

# Base URL for the API (make sure the server is running first)
BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test the various API endpoints"""
    print("Testing RAG FastAPI endpoints...\n")
    
    # Test the root endpoint
    print("1. Testing root endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except requests.exceptions.ConnectionError:
        print("   Server not running. Start with: uvicorn main:app --reload\n")
    
    # Test the health endpoint
    print("2. Testing health endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except requests.exceptions.ConnectionError:
        print("   Server not running. Start with: uvicorn main:app --reload\n")
    
    # Test the documents endpoint
    print("3. Testing documents endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Number of documents: {len(data['documents'])}")
        print(f"   First document: {data['documents'][0]['title']}\n")
    except requests.exceptions.ConnectionError:
        print("   Server not running. Start with: uvicorn main:app --reload\n")
    
    # Test the search endpoint
    print("4. Testing search endpoint:")
    try:
        search_data = {
            "query": "What is RAG?",
            "top_k": 2
        }
        response = requests.post(f"{BASE_URL}/search", 
                                headers={"Content-Type": "application/json"},
                                data=json.dumps(search_data))
        result = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Query: {result['query']}")
        print(f"   Number of relevant documents: {len(result['relevant_documents'])}")
        if result['relevant_documents']:
            print(f"   First match: {result['relevant_documents'][0]['document']['title']}")
            print(f"   Similarity: {result['relevant_documents'][0]['similarity']:.3f}")
        print()
    except requests.exceptions.ConnectionError:
        print("   Server not running. Start with: uvicorn main:app --reload\n")

def show_curl_examples():
    """Show examples of how to use curl to interact with the API"""
    print("Example curl commands:")
    print()
    print("# Get API info")
    print("curl -X GET http://localhost:8000/")
    print()
    print("# Health check")
    print("curl -X GET http://localhost:8000/health")
    print()
    print("# Get all documents")
    print("curl -X GET http://localhost:8000/documents")
    print()
    print("# Search for information")
    print("curl -X POST http://localhost:8000/search \\")
    print("  -H \"Content-Type: application/json\" \\")
    print("  -d '{\"query\": \"What is machine learning?\", \"top_k\": 3}'")
    print()

if __name__ == "__main__":
    test_api_endpoints()
    show_curl_examples()
    print("To run the server: uvicorn main:app --reload")
    print("Then run this script to test the endpoints.")