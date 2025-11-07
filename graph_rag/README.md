# Graph RAG Project

A comprehensive Graph-based Retrieval Augmented Generation (RAG) system built with FastAPI, Neo4j, and LangChain.

## Features

- 📝 **Text Processing**: Automatic cleaning and intelligent chunking of text documents
- 🔍 **Entity Extraction**: NLP-based entity and relationship extraction using spaCy
- 🧮 **Embeddings**: Vector embeddings generation using sentence transformers
- 🗄️ **Knowledge Graph**: Neo4j-based graph database for storing entities, relations, and vectors
- 🔎 **Semantic Search**: Vector similarity search with entity subgraph retrieval
- 🚀 **FastAPI**: RESTful API with comprehensive endpoints
- 🐳 **Docker**: Containerized deployment with Docker Compose

## Architecture

The system follows an object-oriented design with these main components:

1. **TextProcessor**: Cleans and chunks text using LangChain's text splitters
2. **EntityExtractor**: Extracts entities and relationships using spaCy
3. **EmbeddingGenerator**: Creates vector embeddings using HuggingFace sentence transformers
4. **Neo4jGraphStore**: Manages graph database operations with vector indices
5. **GraphRAGPipeline**: Orchestrates the complete pipeline
6. **FastAPI Application**: Provides REST API endpoints

## Pipeline Steps

The Graph RAG pipeline processes documents through these steps:

```
Text File → Cleaning → Chunking → Entity Extraction → 
Relationship Extraction → Embedding Generation → 
Vector Index Creation → Neo4j Storage → Knowledge Graph
```

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

## Quick Start with Docker

### Option 1: Run Both FastAPI and Neo4j

```bash
# Start both services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Run Only Neo4j

```bash
# Start only Neo4j database
docker-compose -f docker-compose-neo4j.yml up -d

# Stop Neo4j
docker-compose -f docker-compose-neo4j.yml down
```

## Local Development Setup

1. **Clone the repository**:
```bash
cd graph_rag
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your Neo4j credentials
```

5. **Start Neo4j** (if not using Docker):
```bash
docker-compose -f docker-compose-neo4j.yml up -d
```

6. **Run the application**:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Process Text File
```bash
curl -X POST "http://localhost:8000/process/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/sample_text.txt"
```

### Process Raw Text
```bash
curl -X POST "http://localhost:8000/process/text" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "\"Your text content here\""
```

### Query the Knowledge Graph
```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "k": 5
  }'
```

### Get Statistics
```bash
curl -X GET "http://localhost:8000/stats"
```

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Clear Database
```bash
curl -X DELETE "http://localhost:8000/clear"
```

## API Documentation

Once the application is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Neo4j Browser

Access Neo4j Browser at http://localhost:7474
- Username: `neo4j`
- Password: `password123` (or your custom password)

### Example Cypher Queries

```cypher
// View all entities
MATCH (e:Entity) RETURN e LIMIT 25

// View all relationships
MATCH (e1:Entity)-[r]->(e2:Entity) RETURN e1, r, e2 LIMIT 25

// View chunks
MATCH (c:Chunk) RETURN c LIMIT 10

// View entity connections
MATCH (e:Entity {text: "Google"})-[r*1..2]-(connected) 
RETURN e, r, connected
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_text_processor.py -v
```

## Project Structure

```
graph_rag/
├── data/                       # Sample data files
│   └── sample_text.txt
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── text_processor.py     # Text cleaning and chunking
│   ├── entity_extractor.py   # Entity and relation extraction
│   ├── embeddings.py          # Embedding generation
│   ├── neo4j_store.py        # Neo4j operations
│   ├── graph_rag_pipeline.py # Pipeline orchestration
│   └── main.py               # FastAPI application
├── tests/                     # Test files
│   ├── __init__.py
│   └── test_text_processor.py
├── docker-compose.yml         # Docker Compose (FastAPI + Neo4j)
├── docker-compose-neo4j.yml  # Docker Compose (Neo4j only)
├── Dockerfile                 # Docker image definition
├── requirements.txt           # Python dependencies
├── pytest.ini                # Pytest configuration
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Configuration

Edit `.env` file to customize:

- **Neo4j Connection**: URI, username, password
- **Embedding Model**: Choose different sentence transformer models
- **Chunk Settings**: Adjust chunk size and overlap
- **API Settings**: Customize API metadata

## Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **Neo4j**: Graph database for knowledge graph storage
- **LangChain**: Framework for text processing and document handling
- **spaCy**: NLP library for entity extraction
- **Sentence Transformers**: Generate semantic embeddings
- **Docker**: Containerization platform
- **pytest**: Testing framework

## Key Features Explained

### 1. Text Cleaning
Removes extra whitespace, special characters, and normalizes text while preserving semantic meaning.

### 2. Intelligent Chunking
Uses LangChain's RecursiveCharacterTextSplitter to create semantically coherent chunks with configurable size and overlap.

### 3. Entity Extraction
Leverages spaCy's NER to identify entities like persons, organizations, locations, and dates.

### 4. Relationship Extraction
Uses dependency parsing to extract subject-verb-object triples and create relationships between entities.

### 5. Vector Search
Creates vector indices in Neo4j for fast similarity search using cosine similarity.

### 6. Knowledge Graph Generation
Stores entities as nodes, relationships as edges, and maintains chunk-to-entity connections.

## Troubleshooting

### Neo4j Connection Issues
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# View Neo4j logs
docker logs graph_rag_neo4j
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Memory Issues
Adjust chunk size in `.env` or use a smaller embedding model.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License

## Acknowledgments

- LangChain for the excellent text processing framework
- Neo4j for the powerful graph database
- spaCy for NLP capabilities
- Sentence Transformers for embedding generation
