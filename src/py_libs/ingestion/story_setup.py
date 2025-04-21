from pathlib import Path
from typing import List, Dict, Any
from .splitter import TextSplitter
from .embedder import TextEmbedder
from .index_builder import IndexBuilder
from .version_manager import VersionManager
from .story_analyzer import StoryAnalyzer
from openai import OpenAI

class StorySetup:
    def __init__(self, story_path: Path, openai_client: OpenAI | None = None):
        """
        Initialize the story setup process.
        
        Args:
            story_path: Path to the story directory
            openai_client: OpenAI client instance to use, if None a new one will be created
        """
        self.story_path = story_path
        self.splitter = TextSplitter()
        self.embedder = TextEmbedder()
        self.index_builder = IndexBuilder()
        self.version_manager = VersionManager(story_path)
        self.analyzer = StoryAnalyzer(story_path, client=openai_client)
        
    def process_story(self, text: str, description: str = "Initial setup") -> str:
        """
        Process a story text through the full ingestion pipeline.
        
        Args:
            text: The story text to process
            description: Description of this version
            
        Returns:
            Version ID of the created index
        """
        # Extract and save story elements
        self.analyzer.update_story_elements(text)
        
        # Split text into chunks
        chunks = self.splitter.split_text(text)
        
        # Embed chunks
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)
        
        # Build index
        self.index_builder.build_index(chunks_with_embeddings)
        
        # Create version first to get the version ID
        version_id = self.version_manager.create_version(description)
        
        # Save artifacts
        self._save_artifacts(chunks_with_embeddings, version_id)
        
        return version_id
        
    def _save_artifacts(self, chunks: List[Dict[str, Any]], version_id: str):
        """
        Save all artifacts to the story directory.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
            version_id: Version ID to save under
        """
        version_dir = self.story_path / "versions" / version_id
        
        # Save chunks with embeddings
        self.embedder.save_embeddings(
            chunks,
            version_dir / "passages.json"
        )
        
        # Save index
        self.index_builder.save_index(
            version_dir / "faiss_index" / "index.faiss"
        )
        
        # Save metadata
        self.index_builder.save_metadata(
            chunks,
            version_dir / "faiss_index" / "metadata.json"
        )
        
    def load_story(self, version_id: str | None = None) -> Dict[str, Any]:
        """
        Load a version of the story.
        
        Args:
            version_id: Version ID to load, if None loads the latest version
            
        Returns:
            Dictionary containing the story data
        """
        if version_id is None:
            version_id = self.version_manager.get_latest_version()
            
        version_dir = self.story_path / "versions" / version_id
        
        return {
            "chunks": self.embedder.load_embeddings(
                version_dir / "passages.json"
            ),
            "index": self.index_builder.load_index(
                version_dir / "faiss_index" / "index.faiss"
            ),
            "metadata": self.index_builder.load_metadata(
                version_dir / "faiss_index" / "metadata.json"
            ),
            "elements": self.analyzer.load_story_elements()
        } 