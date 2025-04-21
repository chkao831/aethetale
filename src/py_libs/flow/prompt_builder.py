from typing import List, Dict, Any
from pathlib import Path
import yaml

class PromptBuilder:
    def __init__(self, story_path: Path):
        """
        Initialize the prompt builder for a story.
        
        Args:
            story_path: Path to the story directory
        """
        self.story_path = story_path
        self.prompt_path = story_path / "prompt.yaml"
        self._load_prompts()
        
    def _load_prompts(self):
        """Load prompt templates from YAML file."""
        with open(self.prompt_path, 'r') as f:
            self.prompts = yaml.safe_load(f)
            
    def build_beat_prompt(self, beat: str, context: List[Dict[str, Any]], style: Dict[str, str]) -> str:
        """
        Build a prompt for expanding a narrative beat.
        
        Args:
            beat: The narrative beat to expand
            context: List of context chunks
            style: Style configuration
            
        Returns:
            Formatted prompt string
        """
        # Format context
        context_text = "\n\n".join([
            f"Context {i+1} (relevance: {chunk['similarity_score']:.2f}):\n{chunk['text']}"
            for i, chunk in enumerate(context)
        ])
        
        # Build prompt
        prompt = self.prompts["beat_expansion"].format(
            beat=beat,
            context=context_text,
            tone=style["tone"],
            pov=style["pov"],
            tense=style["tense"]
        )
        
        return prompt
        
    def build_style_prompt(self, text: str) -> str:
        """
        Build a prompt for style analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Formatted prompt string
        """
        return self.prompts["style_guidance"].format(text=text)
        
    def build_character_prompt(self, character: str, context: List[Dict[str, Any]]) -> str:
        """
        Build a prompt for character development.
        
        Args:
            character: Character name
            context: List of character context chunks
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([
            f"Context {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(context)
        ])
        
        return f"""
        Develop the character {character} based on the following context:
        
        {context_text}
        
        Consider:
        - Personality traits and motivations
        - Relationships with other characters
        - Character arc and development
        - Unique voice and mannerisms
        """
        
    def build_relationship_prompt(self, character1: str, character2: str, context: List[Dict[str, Any]]) -> str:
        """
        Build a prompt for relationship development.
        
        Args:
            character1: First character name
            character2: Second character name
            context: List of relationship context chunks
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([
            f"Context {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(context)
        ])
        
        return f"""
        Develop the relationship between {character1} and {character2} based on the following context:
        
        {context_text}
        
        Consider:
        - Nature of their relationship
        - Key moments in their history
        - Current dynamics and tensions
        - Future potential developments
        """ 