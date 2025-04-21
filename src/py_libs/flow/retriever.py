from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from src.py_libs.ingestion.version_manager import VersionManager

class ContextRetriever:
    def __init__(self, story_path: Path):
        """
        Initialize the context retriever for a story.
        
        Args:
            story_path: Path to the story directory
        """
        self.story_path = story_path
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.version_manager = VersionManager(story_path)
        self._load_index()
        
    def _load_index(self):
        """Load the FAISS index and metadata."""
        # Get current version path
        current_version = self.version_manager.get_current_version()
        if not current_version:
            raise ValueError("No current version found")
            
        version_path = self.story_path / "versions" / current_version
        
        # Load index
        import faiss
        self.index = faiss.read_index(str(version_path / "faiss_index" / "index.faiss"))
        
        # Load metadata
        with open(version_path / "faiss_index" / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
            
    def retrieve_context(self, query: str, num_chunks: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The query string
            num_chunks: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        # Embed query
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            num_chunks
        )
        
        # Get chunks
        chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):  # Ensure index is valid
                chunk = self.metadata[idx].copy()
                chunk['similarity_score'] = float(1 - distance)  # Convert to similarity score
                chunks.append(chunk)
                
        return chunks
        
    def get_character_context(self, character_name: str, num_chunks: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve context specific to a character.
        
        Args:
            character_name: Name of the character
            num_chunks: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        query = f"Character: {character_name}"
        return self.retrieve_context(query, num_chunks)
        
    def get_relationship_context(self, character1: str, character2: str, num_chunks: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve context about the relationship between two characters.
        
        Args:
            character1: Name of the first character
            character2: Name of the second character
            num_chunks: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        query = f"Relationship between {character1} and {character2}"
        return self.retrieve_context(query, num_chunks)
        
    def get_plot_context(self, plot_point: str, num_chunks: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve context about a specific plot point.
        
        Args:
            plot_point: Description of the plot point
            num_chunks: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        query = f"Plot point: {plot_point}"
        return self.retrieve_context(query, num_chunks) 