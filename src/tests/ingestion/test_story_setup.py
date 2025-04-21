import os
import sys
import gc
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import numpy as np
import json
from src.py_libs.ingestion.story_setup import StorySetup

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with predefined story analysis response."""
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''
        {
            "style": {
                "tone": "whimsical",
                "pacing": "steady",
                "narrative_style": "descriptive"
            },
            "characters": {
                "Lena": {
                    "role": "protagonist",
                    "traits": ["curious", "determined"],
                    "arc": "discovery"
                }
            },
            "world": {
                "setting": "magical garden",
                "rules": ["mechanical and natural elements coexist"],
                "atmosphere": "mysterious"
            },
            "themes": ["nature vs technology", "discovery", "harmony"]
        }
        '''))
    ]
    return client

@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns predefined embeddings with all required fields."""
    embedder = Mock()
    
    # Mock the embed_chunks method
    test_embedding = np.random.rand(384)
    embedder.embed_chunks.return_value = [
        {
            "text": "Test chunk",
            "embedding": test_embedding,
            "start_pos": 0,
            "end_pos": 10
        }
    ]
    
    # Mock the save_embeddings method
    def save_embeddings(chunks, output_path):
        # Convert numpy arrays to lists for JSON serialization
        serializable_chunks = []
        for chunk in chunks:
            serializable_chunk = chunk.copy()
            if isinstance(chunk['embedding'], np.ndarray):
                serializable_chunk['embedding'] = chunk['embedding'].tolist()
            serializable_chunks.append(serializable_chunk)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_chunks, f)
    
    embedder.save_embeddings = save_embeddings
    return embedder

def test_story_setup(mock_openai_client, mock_embedder, tmp_path):
    """
    Test the story setup process with mocked dependencies.
    
    This test verifies that:
    1. Story setup initializes correctly
    2. Story analysis works with mocked OpenAI responses
    3. Text splitting and embedding work correctly
    4. Index building succeeds
    5. Version management and artifact saving work
    6. All expected output files are created
    """
    # Create temporary story directory structure
    story_path = tmp_path / "test_story"
    story_path.mkdir()
    (story_path / "content").mkdir()
    (story_path / "content" / "story.txt").write_text("Test story content")
    
    # Create necessary config files
    (story_path / "prompt.yaml").write_text('''
    story_analysis: |
        Analyze the following story text and extract elements:
        {text}
    ''')
    
    # Initialize story setup with mocks
    story_setup = StorySetup(story_path, openai_client=mock_openai_client)
    story_setup.embedder = mock_embedder
    
    # Read and process the story
    with open(story_path / "content" / "story.txt", "r") as f:
        story_text = f.read()
    
    # Test story analysis
    story_setup.analyzer.update_story_elements(story_text)
    assert (story_path / "story_elements.json").exists(), "Story elements file should be created"
    
    # Test text splitting and embedding
    chunks = story_setup.splitter.split_text(story_text)
    assert len(chunks) > 0, "Text should be split into at least one chunk"
    
    chunks_with_embeddings = story_setup.embedder.embed_chunks(chunks)
    assert len(chunks_with_embeddings) > 0, "Embeddings should be generated"
    assert "start_pos" in chunks_with_embeddings[0], "Chunks should have start_pos"
    assert "end_pos" in chunks_with_embeddings[0], "Chunks should have end_pos"
    
    # Test index building
    story_setup.index_builder.build_index(chunks_with_embeddings)
    
    # Test version management and artifact saving
    version_id = story_setup.version_manager.create_version("Initial version")
    story_setup._save_artifacts(chunks_with_embeddings, version_id)
    
    # Verify output files
    version_dir = story_path / "versions" / version_id
    required_files = [
        version_dir / "faiss_index" / "index.faiss",
        version_dir / "faiss_index" / "metadata.json",
        version_dir / "passages.json",
        story_path / "story_elements.json"
    ]
    
    for file_path in required_files:
        assert file_path.exists(), f"Required file {file_path} should exist"

if __name__ == "__main__":
    try:
        test_story_setup()
    except Exception as e:
        print(f"\nTest failed with error: {e}", file=sys.stderr)
        sys.exit(1) 