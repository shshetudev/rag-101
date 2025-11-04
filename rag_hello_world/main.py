"""
Main script to demonstrate the RAG (Retrieval-Augmented Generation) hello world project.
This script shows how to use the RAG system to answer questions based on stored documents.
"""
from rag import SimpleRAG


def main():
    print("=== RAG (Retrieval-Augmented Generation) Hello World ===\n")
    
    # Initialize the RAG system
    rag_system = SimpleRAG()
    
    print("RAG system initialized! The system has been loaded with sample documents.")
    print("You can now ask questions related to these documents.\n")
    
    # Example queries to demonstrate the system
    example_queries = [
        "What is RAG?",
        "Tell me about machine learning",
        "Explain natural language processing",
        "How are vector embeddings used?",
        "What is Python programming?"
    ]
    
    print("=== Example Queries ===")
    for i, query in enumerate(example_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        # Get response from RAG system
        result = rag_system.query(query)
        
        # Print the response
        print(f"Response: {result['response'].split('Based on the provided')[0]}")
        
        # Show which documents were retrieved
        print("Retrieved documents:")
        for j, doc in enumerate(result['relevant_documents'], 1):
            print(f"  {j}. {doc['document']['title']} (similarity: {doc['similarity']:.3f})")
    
    print("\n=== Interactive Mode ===")
    print("Now you can ask your own questions! Type 'quit' to exit.\n")
    
    while True:
        user_query = input("Your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Thanks for trying the RAG hello world project!")
            break
        
        if not user_query:
            continue
        
        # Get response from RAG system
        result = rag_system.query(user_query)
        
        # Print the response
        print(f"\nRAG Response: {result['response']}")
        print()
        
        # Show which documents were retrieved with their similarity scores
        print("Documents used for response:")
        for i, doc in enumerate(result['relevant_documents'], 1):
            print(f"  {i}. {doc['document']['title']} (similarity: {doc['similarity']:.3f})")
        print("-" * 60)


if __name__ == "__main__":
    main()