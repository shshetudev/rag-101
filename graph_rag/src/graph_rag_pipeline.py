"""Graph RAG pipeline orchestration."""
from typing import List, Dict, Any
from langchain.schema import Document
from src.config import settings
from src.text_processor import TextProcessor
from src.entity_extractor import EntityExtractor, Entity, Relation
from src.embeddings import EmbeddingGenerator
from src.neo4j_store import Neo4jGraphStore


class GraphRAGPipeline:
    """Orchestrates the complete Graph RAG pipeline."""
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None
    ):
        """
        Initialize the Graph RAG pipeline.
        
        Args:
            neo4j_uri: Neo4j connection URI (optional, defaults to settings)
            neo4j_user: Neo4j username (optional, defaults to settings)
            neo4j_password: Neo4j password (optional, defaults to settings)
        """
        self.text_processor = TextProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.entity_extractor = EntityExtractor()
        self.embedding_generator = EmbeddingGenerator(
            model_name=settings.embedding_model_name
        )
        self.graph_store = Neo4jGraphStore(
            uri=neo4j_uri or settings.neo4j_uri,
            user=neo4j_user or settings.neo4j_user,
            password=neo4j_password or settings.neo4j_password
        )
    
    def process_text_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a text file through the complete pipeline.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with processing statistics
        """
        # Step 1: Clean and chunk text
        documents = self.text_processor.process_text_file(file_path)
        
        # Step 2: Extract entities and relations
        entities, relations = self.entity_extractor.process_documents(documents)
        
        # Step 3: Generate embeddings
        embeddings = self.embedding_generator.generate_document_embeddings(documents)
        
        # Step 4: Initialize database
        self.graph_store.create_constraints()
        embedding_dim = self.embedding_generator.get_embedding_dimension()
        self.graph_store.create_vector_index(dimension=embedding_dim)
        
        # Step 5: Store in Neo4j
        entity_count = self.graph_store.store_entities(entities)
        relation_count = self.graph_store.store_relations(relations)
        chunk_count = self.graph_store.store_chunks_with_embeddings(documents, embeddings)
        
        # Step 6: Link chunks to entities
        self.graph_store.link_chunks_to_entities(documents, entities)
        
        return {
            "file_path": file_path,
            "chunks_processed": chunk_count,
            "entities_extracted": entity_count,
            "relations_extracted": relation_count,
            "embedding_dimension": embedding_dim,
            "status": "success"
        }
    
    def query(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge graph.
        
        Args:
            query_text: Query text
            k: Number of results to return
            
        Returns:
            Dictionary with query results
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        
        # Perform similarity search
        results = self.graph_store.similarity_search(query_embedding, k=k)
        
        # Extract entities from query
        query_entities = self.entity_extractor.extract_entities(query_text)
        
        # Get subgraphs for query entities
        subgraphs = []
        for entity in query_entities[:3]:  # Limit to top 3 entities
            subgraph = self.graph_store.get_entity_subgraph(entity.text)
            if subgraph["nodes"]:
                subgraphs.append({
                    "entity": entity.text,
                    "subgraph": subgraph
                })
        
        return {
            "query": query_text,
            "similar_chunks": results,
            "query_entities": [{"text": e.text, "label": e.label} for e in query_entities],
            "entity_subgraphs": subgraphs
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get knowledge graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        return self.graph_store.get_statistics()
    
    def clear_graph(self):
        """Clear the knowledge graph."""
        self.graph_store.clear_database()
    
    def close(self):
        """Close database connections."""
        self.graph_store.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
