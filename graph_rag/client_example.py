"""Example client for testing the Graph RAG API."""
import requests
import json
from pathlib import Path


class GraphRAGClient:
    """Client for interacting with the Graph RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
    
    def health_check(self):
        """Check API health status."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def process_file(self, file_path: str):
        """
        Process a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Processing results
        """
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'text/plain')}
            response = requests.post(f"{self.base_url}/process/file", files=files)
        return response.json()
    
    def process_text(self, text: str):
        """
        Process raw text.
        
        Args:
            text: Raw text content
            
        Returns:
            Processing results
        """
        response = requests.post(
            f"{self.base_url}/process/text",
            params={"text": text}
        )
        return response.json()
    
    def query(self, query_text: str, k: int = 5):
        """
        Query the knowledge graph.
        
        Args:
            query_text: Query text
            k: Number of results
            
        Returns:
            Query results
        """
        payload = {
            "query": query_text,
            "k": k
        }
        response = requests.post(
            f"{self.base_url}/query",
            json=payload
        )
        return response.json()
    
    def get_stats(self):
        """Get knowledge graph statistics."""
        response = requests.get(f"{self.base_url}/stats")
        return response.json()
    
    def clear_graph(self):
        """Clear the knowledge graph."""
        response = requests.delete(f"{self.base_url}/clear")
        return response.json()


def main():
    """Example usage of the Graph RAG client."""
    client = GraphRAGClient()
    
    print("=" * 60)
    print("Graph RAG API Client Example")
    print("=" * 60)
    
    # 1. Health check
    print("\n1. Checking API health...")
    try:
        health = client.health_check()
        print(f"✓ API Status: {health.get('status')}")
        print(f"✓ Database: {health.get('database')}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # 2. Process sample file
    print("\n2. Processing sample text file...")
    try:
        sample_file = "data/sample_text.txt"
        result = client.process_file(sample_file)
        print(f"✓ Chunks processed: {result.get('chunks_processed')}")
        print(f"✓ Entities extracted: {result.get('entities_extracted')}")
        print(f"✓ Relations extracted: {result.get('relations_extracted')}")
        print(f"✓ Embedding dimension: {result.get('embedding_dimension')}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 3. Get statistics
    print("\n3. Getting knowledge graph statistics...")
    try:
        stats = client.get_stats()
        print(f"✓ Total entities: {stats.get('entities')}")
        print(f"✓ Total chunks: {stats.get('chunks')}")
        print(f"✓ Total relationships: {stats.get('relationships')}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 4. Query the knowledge graph
    print("\n4. Querying the knowledge graph...")
    queries = [
        "What is artificial intelligence?",
        "Tell me about Google and machine learning",
        "What are knowledge graphs?"
    ]
    
    for query_text in queries:
        print(f"\n   Query: '{query_text}'")
        try:
            result = client.query(query_text, k=3)
            print(f"   ✓ Found {len(result.get('similar_chunks', []))} similar chunks")
            print(f"   ✓ Extracted {len(result.get('query_entities', []))} entities from query")
            
            # Show top result
            if result.get('similar_chunks'):
                top_chunk = result['similar_chunks'][0]
                preview = top_chunk.get('text', '')[:100] + "..."
                print(f"   ✓ Top result preview: {preview}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # 5. Final statistics
    print("\n5. Final statistics...")
    try:
        stats = client.get_stats()
        print(f"✓ Knowledge graph contains:")
        print(f"  - {stats.get('entities')} entities")
        print(f"  - {stats.get('chunks')} text chunks")
        print(f"  - {stats.get('relationships')} relationships")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed! Visit http://localhost:8000/docs for API documentation")
    print("=" * 60)


if __name__ == "__main__":
    main()
