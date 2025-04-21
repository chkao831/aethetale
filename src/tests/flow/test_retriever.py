import pytest
from pathlib import Path
import json
import numpy as np
import faiss
from src.py_libs.flow.retriever import ContextRetriever
from src.py_libs.ingestion.version_manager import VersionManager

@pytest.fixture
def temp_story_dir(tmp_path):
    """Create a temporary story directory with test index."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    # Initialize version manager
    version_manager = VersionManager(story_dir)
    version_id = version_manager.create_version("Test version")
    version_path = story_dir / "versions" / version_id
    
    # Create FAISS index directory
    index_dir = version_path / "faiss_index"
    index_dir.mkdir(parents=True)
    
    # Create test metadata
    metadata = [
        {"text": "Test context 1", "type": "character", "character": "Alice"},
        {"text": "Test context 2", "type": "relationship", "characters": ["Alice", "Bob"]},
        {"text": "Test context 3", "type": "plot", "plot_point": "introduction"}
    ]
    
    # Create a simple FAISS index
    dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
    index = faiss.IndexFlatL2(dimension)
    # Add some random vectors
    num_vectors = len(metadata)
    vectors = np.random.random((num_vectors, dimension)).astype('float32')
    index.add(vectors)
    
    # Save the index
    faiss.write_index(index, str(index_dir / "index.faiss"))
    
    # Save metadata
    with open(index_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return story_dir

def test_retriever_initialization(temp_story_dir):
    """Test ContextRetriever initialization."""
    retriever = ContextRetriever(temp_story_dir)
    assert retriever.story_path == temp_story_dir
    assert hasattr(retriever, "embedder")
    assert hasattr(retriever, "index")
    assert hasattr(retriever, "metadata")

def test_retrieve_context(temp_story_dir):
    """Test context retrieval."""
    retriever = ContextRetriever(temp_story_dir)
    query = "Test query"
    chunks = retriever.retrieve_context(query, num_chunks=2)
    
    assert len(chunks) <= 2
    for chunk in chunks:
        assert "text" in chunk
        assert "similarity_score" in chunk
        assert isinstance(chunk["similarity_score"], float)

def test_get_character_context(temp_story_dir):
    """Test character context retrieval."""
    retriever = ContextRetriever(temp_story_dir)
    chunks = retriever.get_character_context("Alice", num_chunks=1)
    
    assert len(chunks) <= 1
    if chunks:
        assert "text" in chunks[0]
        assert "similarity_score" in chunks[0]

def test_get_relationship_context(temp_story_dir):
    """Test relationship context retrieval."""
    retriever = ContextRetriever(temp_story_dir)
    chunks = retriever.get_relationship_context("Alice", "Bob", num_chunks=1)
    
    assert len(chunks) <= 1
    if chunks:
        assert "text" in chunks[0]
        assert "similarity_score" in chunks[0]

def test_get_plot_context(temp_story_dir):
    """Test plot context retrieval."""
    retriever = ContextRetriever(temp_story_dir)
    chunks = retriever.get_plot_context("introduction", num_chunks=1)
    
    assert len(chunks) <= 1
    if chunks:
        assert "text" in chunks[0]
        assert "similarity_score" in chunks[0] 