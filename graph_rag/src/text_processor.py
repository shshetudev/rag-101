"""Text processing utilities for cleaning and chunking text."""
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class TextProcessor:
    """Handles text cleaning and chunking operations."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing extra whitespace and special characters.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        
        # Remove multiple consecutive punctuation marks
        text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_into_chunks(self, text: str) -> List[Document]:
        """
        Split text into chunks using LangChain's text splitter.
        
        Args:
            text: Cleaned text to split
            
        Returns:
            List of Document objects containing text chunks
        """
        documents = self.text_splitter.create_documents([text])
        return documents
    
    def process_text_file(self, file_path: str) -> List[Document]:
        """
        Process a text file: read, clean, and chunk.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        cleaned_text = self.clean_text(raw_text)
        chunks = self.split_into_chunks(cleaned_text)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                "source": file_path,
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            }
        
        return chunks
