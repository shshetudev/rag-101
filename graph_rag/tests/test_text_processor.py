"""Unit tests for TextProcessor class."""
import pytest
from src.text_processor import TextProcessor


class TestTextProcessor:
    """Test suite for TextProcessor class."""
    
    @pytest.fixture
    def text_processor(self):
        """Create a TextProcessor instance for testing."""
        return TextProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_clean_text_removes_extra_whitespace(self, text_processor):
        """Test that clean_text removes extra whitespace."""
        # Arrange
        input_text = "This  is   a    test   with     extra    whitespace."
        expected = "This is a test with extra whitespace."
        
        # Act
        result = text_processor.clean_text(input_text)
        
        # Assert
        assert result == expected
        assert "  " not in result  # No double spaces
    
    def test_clean_text_removes_special_characters(self, text_processor):
        """Test that clean_text removes special characters."""
        # Arrange
        input_text = "Hello @#$% World! Testing & cleaning."
        
        # Act
        result = text_processor.clean_text(input_text)
        
        # Assert
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result
        assert "&" not in result
        # Basic punctuation should remain
        assert "!" in result
    
    def test_clean_text_handles_multiple_punctuation(self, text_processor):
        """Test that clean_text handles multiple consecutive punctuation marks."""
        # Arrange
        input_text = "What???!!! Is this... really necessary......"
        
        # Act
        result = text_processor.clean_text(input_text)
        
        # Assert
        # Should reduce multiple punctuation to single
        assert "???" not in result
        assert "..." not in result or result.count("...") == 1
    
    def test_clean_text_preserves_basic_punctuation(self, text_processor):
        """Test that clean_text preserves basic punctuation."""
        # Arrange
        input_text = "Hello, world! How are you? I'm fine."
        
        # Act
        result = text_processor.clean_text(input_text)
        
        # Assert
        assert "," in result
        assert "!" in result
        assert "?" in result
        assert "'" in result
    
    def test_split_into_chunks_returns_documents(self, text_processor):
        """Test that split_into_chunks returns Document objects."""
        # Arrange
        input_text = "This is a test sentence. " * 20  # Create longer text
        
        # Act
        chunks = text_processor.split_into_chunks(input_text)
        
        # Assert
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'page_content') for chunk in chunks)
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)
    
    def test_split_into_chunks_respects_chunk_size(self, text_processor):
        """Test that chunks respect the configured chunk size."""
        # Arrange
        input_text = "This is a test sentence. " * 50
        
        # Act
        chunks = text_processor.split_into_chunks(input_text)
        
        # Assert
        for chunk in chunks:
            # Each chunk should be approximately chunk_size or smaller
            assert len(chunk.page_content) <= text_processor.chunk_size + 50  # Allow some tolerance
    
    def test_process_text_file_adds_metadata(self, text_processor, tmp_path):
        """Test that process_text_file adds metadata to chunks."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_content = "This is test content. " * 20
        test_file.write_text(test_content)
        
        # Act
        chunks = text_processor.process_text_file(str(test_file))
        
        # Assert
        assert len(chunks) > 0
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "chunk_size" in chunk.metadata
            assert chunk.metadata["source"] == str(test_file)
    
    def test_process_text_file_cleans_text(self, text_processor, tmp_path):
        """Test that process_text_file cleans the text."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_content = "This   has    extra     spaces  and @#$ special chars."
        test_file.write_text(test_content)
        
        # Act
        chunks = text_processor.process_text_file(str(test_file))
        
        # Assert
        combined_text = " ".join([chunk.page_content for chunk in chunks])
        assert "  " not in combined_text  # No double spaces
        assert "@" not in combined_text
        assert "#" not in combined_text
    
    def test_chunk_overlap_creates_continuity(self, text_processor):
        """Test that chunk overlap creates continuity between chunks."""
        # Arrange
        sentences = [f"Sentence number {i}. " for i in range(20)]
        input_text = "".join(sentences)
        
        # Act
        chunks = text_processor.split_into_chunks(input_text)
        
        # Assert
        if len(chunks) > 1:
            # With overlap, some content should appear in consecutive chunks
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i].page_content
                next_chunk = chunks[i + 1].page_content
                # Check if there's any word overlap
                current_words = set(current_chunk.split()[-10:])  # Last 10 words
                next_words = set(next_chunk.split()[:10])  # First 10 words
                # There should be some overlap
                assert len(current_words & next_words) > 0 or text_processor.chunk_overlap == 0
