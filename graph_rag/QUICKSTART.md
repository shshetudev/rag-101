# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Start the Services

**Option A: Full Stack (Recommended for first-time users)**
```bash
cd graph_rag
docker-compose up -d
```

**Option B: Neo4j Only (for local development)**
```bash
cd graph_rag
docker-compose -f docker-compose-neo4j.yml up -d
```

### Step 2: Verify Services are Running

Check the health endpoint:
```bash
curl http://localhost:8000/health
```

Or visit in your browser:
- API Documentation: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474 (username: `neo4j`, password: `password123`)

### Step 3: Process Your First Document

Using the provided sample file:
```bash
curl -X POST "http://localhost:8000/process/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/sample_text.txt"
```

Or use the Python client:
```bash
python client_example.py
```

## 📊 What Happens Behind the Scenes?

When you process a document, the system:

1. **Cleans** the text (removes extra spaces, special characters)
2. **Chunks** it into manageable pieces (default: 500 chars with 50 char overlap)
3. **Extracts entities** (people, organizations, locations, etc.) using spaCy
4. **Finds relationships** between entities using NLP
5. **Generates embeddings** using sentence transformers (384-dimensional vectors)
6. **Creates vector index** in Neo4j for fast similarity search
7. **Stores everything** in the graph database:
   - Entities as nodes
   - Relationships as edges
   - Text chunks with embeddings
   - Links between chunks and entities

## 🔍 Query the Knowledge Graph

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "k": 5
  }'
```

The query will:
- Convert your question to an embedding
- Find similar text chunks using vector search
- Extract entities from your query
- Return relevant subgraphs showing connections

## 📈 View Statistics

```bash
curl http://localhost:8000/stats
```

## 🧪 Run Tests

```bash
# Local environment
python -m pytest -v

# With coverage report
python -m pytest --cov=src --cov-report=html
```

## 🛑 Stop Services

```bash
docker-compose down
```

## 🔧 Troubleshooting

### Services won't start?
```bash
# Check what's running
docker ps

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

### Connection refused?
Wait 10-15 seconds after starting services for Neo4j to fully initialize.

### Import errors in tests?
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 🎯 Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Try different queries**: Test various questions
3. **Upload your own files**: Process your documents
4. **Visualize in Neo4j**: Browse the knowledge graph at http://localhost:7474
5. **Customize**: Edit `.env` to change chunk size, model, etc.

## 📚 Learn More

- See `README.md` for complete documentation
- Check `src/` folder for implementation details
- Read `tests/` for usage examples
- Explore Neo4j with Cypher queries in the browser

## 💡 Example Workflow

```python
# Using Python
from src.graph_rag_pipeline import GraphRAGPipeline

# Initialize
pipeline = GraphRAGPipeline()

# Process a document
result = pipeline.process_text_file("data/sample_text.txt")
print(f"Processed {result['chunks_processed']} chunks")
print(f"Found {result['entities_extracted']} entities")

# Query
query_result = pipeline.query("Tell me about AI", k=5)
for chunk in query_result['similar_chunks']:
    print(f"Score: {chunk['score']:.3f}")
    print(f"Text: {chunk['text'][:100]}...")

# Get stats
stats = pipeline.get_statistics()
print(f"Graph has {stats['entities']} entities and {stats['relationships']} relationships")

# Cleanup
pipeline.close()
```

Happy Graph RAG-ing! 🎉
