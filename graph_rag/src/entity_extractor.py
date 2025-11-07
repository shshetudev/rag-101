"""Entity and relationship extraction using spaCy and LangChain."""
import spacy
from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain.schema import Document


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int


@dataclass
class Relation:
    """Represents a relationship between entities."""
    source: str
    target: str
    relation_type: str
    context: str


class EntityExtractor:
    """Extracts entities and relationships from text using spaCy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Model not found, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            ))
        
        return entities
    
    def extract_relations(self, text: str) -> List[Relation]:
        """
        Extract relationships between entities using dependency parsing.
        
        Args:
            text: Input text
            
        Returns:
            List of Relation objects
        """
        doc = self.nlp(text)
        relations = []
        
        # Extract subject-verb-object triples
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    subject = token.text
                    verb = token.head.text
                    
                    # Find object
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "attr", "pobj"):
                            obj = child.text
                            relations.append(Relation(
                                source=subject,
                                target=obj,
                                relation_type=verb,
                                context=sent.text
                            ))
        
        return relations
    
    def extract_entity_relations(self, text: str) -> List[Relation]:
        """
        Extract relationships specifically between named entities.
        
        Args:
            text: Input text
            
        Returns:
            List of Relation objects between named entities
        """
        doc = self.nlp(text)
        entities = {ent.text: ent.label_ for ent in doc.ents}
        relations = []
        
        for sent in doc.sents:
            sent_entities = [ent for ent in sent.ents]
            
            # Create relations between entities in the same sentence
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    # Find the verb connecting them
                    verb = self._find_connecting_verb(sent, ent1, ent2)
                    
                    relations.append(Relation(
                        source=ent1.text,
                        target=ent2.text,
                        relation_type=verb if verb else "RELATED_TO",
                        context=sent.text
                    ))
        
        return relations
    
    def _find_connecting_verb(self, sent, ent1, ent2) -> str:
        """Find the verb connecting two entities in a sentence."""
        # Simple heuristic: find verbs between the entities
        verbs = [token.text for token in sent if token.pos_ == "VERB"]
        return verbs[0] if verbs else "RELATED_TO"
    
    def process_documents(self, documents: List[Document]) -> Tuple[List[Entity], List[Relation]]:
        """
        Process multiple documents to extract entities and relations.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of (entities, relations)
        """
        all_entities = []
        all_relations = []
        
        for doc in documents:
            entities = self.extract_entities(doc.page_content)
            relations = self.extract_entity_relations(doc.page_content)
            
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        return all_entities, all_relations
