from pathlib import Path
import json
import yaml
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, story_path: Path):
        """
        Initialize the config loader for a story.
        
        Args:
            story_path: Path to the story directory
        """
        self.story_path = story_path
        self.config_path = story_path / "config.json"
        self.prompt_path = story_path / "prompt.yaml"
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load the story configuration.
        
        Returns:
            Dictionary containing the story configuration
        """
        if not self.config_path.exists():
            return self._create_default_config()
            
        with open(self.config_path, 'r') as f:
            return json.load(f)
            
    def load_prompts(self) -> Dict[str, Any]:
        """
        Load the prompt templates.
        
        Returns:
            Dictionary containing the prompt templates
        """
        if not self.prompt_path.exists():
            return self._create_default_prompts()
            
        with open(self.prompt_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default story configuration."""
        config = {
            "story_name": self.story_path.name,
            "model": {
                "name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "retrieval": {
                "num_chunks": 5,
                "similarity_threshold": 0.7
            },
            "style": {
                "tone": "neutral",
                "pov": "third_person",
                "tense": "past"
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return config
        
    def _create_default_prompts(self) -> Dict[str, Any]:
        """Create default prompt templates."""
        prompts = {
            "beat_expansion": """
            Expand the following narrative beat into a detailed scene:
            
            Beat: {beat}
            
            Context from previous story:
            {context}
            
            Style guidelines:
            - Tone: {tone}
            - Point of view: {pov}
            - Tense: {tense}
            
            Write a detailed scene that:
            1. Maintains consistency with the provided context
            2. Follows the style guidelines
            3. Develops the characters and their relationships
            4. Advances the plot naturally
            """,
            "style_guidance": """
            Analyze the following text and provide style guidance:
            
            Text: {text}
            
            Consider:
            - Tone and mood
            - Narrative voice
            - Character voice consistency
            - Pacing and structure
            """
        }
        
        with open(self.prompt_path, 'w') as f:
            yaml.dump(prompts, f)
            
        return prompts 