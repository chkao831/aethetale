from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from src.py_libs.ingestion.version_manager import VersionManager
from src.py_libs.models.character_profile import CharacterProfile

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
        self._load_character_profiles()
        
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
            
    def _load_character_profiles(self):
        """Load character profiles from the story directory."""
        profiles_path = self.story_path / "character_profiles.json"
        if profiles_path.exists():
            with open(profiles_path, 'r') as f:
                profiles_data = json.load(f)
                self.character_profiles = {
                    name: CharacterProfile.from_dict(data)
                    for name, data in profiles_data.items()
                }
        else:
            self.character_profiles = {}
            
    def _save_character_profiles(self):
        """Save character profiles to disk."""
        profiles_path = self.story_path / "character_profiles.json"
        profiles_data = {
            name: profile.to_dict()
            for name, profile in self.character_profiles.items()
        }
        with open(profiles_path, 'w') as f:
            json.dump(profiles_data, f, indent=2)
            
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
        
    def get_character_profile(self, character_name: str) -> Optional[CharacterProfile]:
        """
        Get a character's full profile.
        
        Args:
            character_name: Name of the character
            
        Returns:
            CharacterProfile object if found, None otherwise
        """
        # Try exact match first
        if character_name in self.character_profiles:
            return self.character_profiles[character_name]
            
        # Try aliases
        for profile in self.character_profiles.values():
            if character_name in profile.aliases:
                return profile
                
        return None
        
    def get_character_context(self, character_name: str, num_chunks: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve context specific to a character.
        
        Args:
            character_name: Name of the character
            num_chunks: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        # Get character profile
        profile = self.get_character_profile(character_name)
        if not profile:
            # Fallback to basic retrieval if no profile exists
            query = f"Character: {character_name}"
            return self.retrieve_context(query, num_chunks)
            
        # Build rich query using profile information
        query_parts = [
            f"Character: {character_name}",
            f"Role: {profile.role}",
            f"Traits: {', '.join(profile.personality_traits)}",
            f"Goals: {', '.join(profile.goals)}"
        ]
        query = " ".join(query_parts)
        
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
        # Get character profiles
        profile1 = self.get_character_profile(character1)
        profile2 = self.get_character_profile(character2)
        
        # Build relationship query
        query_parts = [f"Relationship between {character1} and {character2}"]
        
        if profile1 and profile2:
            # Add relationship-specific information
            if character2 in profile1.lovers:
                query_parts.append("romantic relationship")
            if character2 in profile1.friends:
                query_parts.append("friendship")
            if character2 in profile1.enemies:
                query_parts.append("conflict")
            if character2 in [f["name"] for f in profile1.family]:
                query_parts.append("family relationship")
                
        query = " ".join(query_parts)
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
        
    def get_characters_by_trait(self, trait: str) -> List[str]:
        """
        Get all characters that have a specific trait.
        
        Args:
            trait: The trait to search for
            
        Returns:
            List of character names
        """
        matching_characters = []
        for name, profile in self.character_profiles.items():
            if trait.lower() in [t.lower() for t in profile.personality_traits]:
                matching_characters.append(name)
        return matching_characters
        
    def get_characters_by_relationship(self, character_name: str, relationship_type: str) -> List[str]:
        """
        Get all characters that have a specific relationship with a character.
        
        Args:
            character_name: Name of the character
            relationship_type: Type of relationship (e.g., 'friends', 'enemies', 'lovers')
            
        Returns:
            List of character names
        """
        profile = self.get_character_profile(character_name)
        if not profile:
            return []
            
        if relationship_type == 'friends':
            return profile.friends
        elif relationship_type == 'enemies':
            return profile.enemies
        elif relationship_type == 'lovers':
            return profile.lovers
        elif relationship_type == 'family':
            return [f["name"] for f in profile.family]
        else:
            return [] 