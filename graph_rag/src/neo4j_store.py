"""Neo4j database operations for storing and querying the knowledge graph."""
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from langchain.schema import Document
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from src.entity_extractor import Entity, Relation


class Neo4jGraphStore:
    """Manages Neo4j graph database operations."""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the Neo4j graph store.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Initialize LangChain Neo4j graph
        self.graph = Neo4jGraph(
            url=uri,
            username=user,
            password=password
        )
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def create_constraints(self):
        """Create uniqueness constraints for entities."""
        with self.driver.session() as session:
            # Create constraint for Entity nodes
            session.run("""
                CREATE CONSTRAINT entity_text IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.text IS UNIQUE
            """)
            
            # Create constraint for Chunk nodes
            session.run("""
                CREATE CONSTRAINT chunk_id IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
            """)
    
    def create_vector_index(self, index_name: str = "chunk_embeddings", dimension: int = 384):
        """
        Create a vector index for similarity search.
        
        Args:
            index_name: Name of the vector index
            dimension: Dimension of the embedding vectors
        """
        with self.driver.session() as session:
            # Drop existing index if it exists
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            
            # Create new vector index
            session.run(f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (c:Chunk)
                ON c.embedding
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dimension},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)
    
    def store_entities(self, entities: List[Entity]) -> int:
        """
        Store entities as nodes in the graph.
        
        Args:
            entities: List of Entity objects
            
        Returns:
            Number of entities stored
        """
        with self.driver.session() as session:
            count = 0
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {text: $text})
                    SET e.label = $label,
                        e.start = $start,
                        e.end = $end
                """, text=entity.text, label=entity.label, 
                   start=entity.start, end=entity.end)
                count += 1
            return count
    
    def store_relations(self, relations: List[Relation]) -> int:
        """
        Store relationships between entities.
        
        Args:
            relations: List of Relation objects
            
        Returns:
            Number of relations stored
        """
        with self.driver.session() as session:
            count = 0
            for relation in relations:
                # Create a safe relationship type (replace spaces and special chars)
                safe_rel_type = relation.relation_type.upper().replace(" ", "_").replace("-", "_")
                if not safe_rel_type or not safe_rel_type[0].isalpha():
                    safe_rel_type = "RELATED_TO"
                
                session.run(f"""
                    MATCH (source:Entity {{text: $source}})
                    MATCH (target:Entity {{text: $target}})
                    MERGE (source)-[r:{safe_rel_type}]->(target)
                    SET r.context = $context
                """, source=relation.source, target=relation.target, 
                   context=relation.context)
                count += 1
            return count
    
    def store_chunks_with_embeddings(
        self, 
        documents: List[Document], 
        embeddings: List[List[float]]
    ) -> int:
        """
        Store text chunks with their embeddings.
        
        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
            
        Returns:
            Number of chunks stored
        """
        with self.driver.session() as session:
            count = 0
            for doc, embedding in zip(documents, embeddings):
                chunk_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', count)}"
                
                session.run("""
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.text = $text,
                        c.embedding = $embedding,
                        c.source = $source,
                        c.chunk_size = $chunk_size
                """, chunk_id=chunk_id, text=doc.page_content, 
                   embedding=embedding, source=doc.metadata.get('source', 'unknown'),
                   chunk_size=doc.metadata.get('chunk_size', len(doc.page_content)))
                count += 1
            return count
    
    def link_chunks_to_entities(self, documents: List[Document], entities: List[Entity]):
        """
        Create relationships between chunks and entities mentioned in them.
        
        Args:
            documents: List of Document objects
            entities: List of Entity objects
        """
        with self.driver.session() as session:
            for i, doc in enumerate(documents):
                chunk_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', i)}"
                text = doc.page_content
                
                # Find entities mentioned in this chunk
                for entity in entities:
                    if entity.text in text:
                        session.run("""
                            MATCH (c:Chunk {chunk_id: $chunk_id})
                            MATCH (e:Entity {text: $entity_text})
                            MERGE (c)-[:MENTIONS]->(e)
                        """, chunk_id=chunk_id, entity_text=entity.text)
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using vector index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
                YIELD node, score
                RETURN node.chunk_id AS chunk_id, 
                       node.text AS text, 
                       node.source AS source,
                       score
                ORDER BY score DESC
            """, k=k, query_embedding=query_embedding)
            
            return [dict(record) for record in result]
    
    def get_entity_subgraph(self, entity_text: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get subgraph around an entity.
        
        Args:
            entity_text: Text of the entity
            depth: Depth of the subgraph traversal
            
        Returns:
            Dictionary containing nodes and relationships
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (e:Entity {text: $entity_text})-[*1..%d]-(connected)
                RETURN path
            """ % depth, entity_text=entity_text)
            
            nodes = set()
            relationships = []
            
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    nodes.add((node.get("text"), dict(node)))
                for rel in path.relationships:
                    relationships.append({
                        "type": rel.type,
                        "start": rel.start_node.get("text"),
                        "end": rel.end_node.get("text")
                    })
            
            return {
                "nodes": [{"text": n[0], "properties": n[1]} for n in nodes],
                "relationships": relationships
            }
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with counts of nodes and relationships
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WITH count(e) as entity_count
                MATCH (c:Chunk)
                WITH entity_count, count(c) as chunk_count
                MATCH ()-[r]->()
                RETURN entity_count, chunk_count, count(r) as relationship_count
            """)
            
            record = result.single()
            if record:
                return {
                    "entities": record["entity_count"],
                    "chunks": record["chunk_count"],
                    "relationships": record["relationship_count"]
                }
            return {"entities": 0, "chunks": 0, "relationships": 0}
