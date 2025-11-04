"""
Sample documents for our RAG FastAPI project.
This represents our knowledge base that we'll query against.
"""
DOCUMENTS = [
    {
        "id": 1,
        "title": "Introduction to RAG",
        "content": "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation to provide more accurate and contextually relevant responses."
    },
    {
        "id": 2,
        "title": "Machine Learning Basics",
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
    },
    {
        "id": 3,
        "title": "Natural Language Processing",
        "content": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language."
    },
    {
        "id": 4,
        "title": "Vector Embeddings",
        "content": "Vector embeddings are numerical representations of text that capture semantic meaning. They allow us to measure similarity between different pieces of text."
    },
    {
        "id": 5,
        "title": "Python Programming",
        "content": "Python is a high-level, interpreted programming language known for its simplicity and wide range of applications in web development, data science, and AI."
    }
]