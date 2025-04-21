from typing import Dict, Any
import json
from openai import OpenAI
from pathlib import Path

class StoryAnalyzer:
    def __init__(self, story_path: Path, client: OpenAI | None = None):
        """
        Initialize the story analyzer.
        
        Args:
            story_path: Path to the story directory
            client: OpenAI client instance, if None a new one will be created
        """
        self.story_path = story_path
        self.prompt_path = story_path / "prompt.yaml"
        self.client = client if client is not None else OpenAI()
        
    def extract_story_elements(self, text: str) -> Dict[str, Any]:
        """
        Extract story elements from the text using LLM.
        
        Args:
            text: The story text to analyze
            
        Returns:
            Dictionary containing extracted story elements
        """
        # Load the story analysis prompt
        with open(self.prompt_path, 'r') as f:
            import yaml
            prompts = yaml.safe_load(f)
            analysis_prompt = prompts['story_analysis']
            
        # Remove indentation from the template
        analysis_prompt = '\n'.join(line.strip() for line in analysis_prompt.split('\n'))
        
        # Format the prompt
        formatted_prompt = analysis_prompt.format(text=text)
        
        # Call the LLM
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a story analysis assistant."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.3
        )
        
        # Parse the response
        try:
            elements = json.loads(response.choices[0].message.content)
            return elements
        except json.JSONDecodeError:
            raise ValueError("Failed to parse story elements from LLM response")
            
    def save_story_elements(self, elements: Dict[str, Any]):
        """
        Save extracted story elements to a file.
        
        Args:
            elements: Dictionary of story elements
        """
        with open(self.story_path / "story_elements.json", 'w') as f:
            json.dump(elements, f, indent=2)
            
    def load_story_elements(self) -> Dict[str, Any]:
        """
        Load story elements from file.
        
        Returns:
            Dictionary of story elements
        """
        with open(self.story_path / "story_elements.json", 'r') as f:
            return json.load(f)
            
    def update_story_elements(self, new_text: str):
        """
        Update story elements with new text.
        
        Args:
            new_text: New text to analyze
        """
        # Load existing elements
        try:
            existing_elements = self.load_story_elements()
        except FileNotFoundError:
            existing_elements = {}
            
        # Extract elements from new text
        new_elements = self.extract_story_elements(new_text)
        
        # Merge elements
        merged_elements = self._merge_elements(existing_elements, new_elements)
        
        # Save updated elements
        self.save_story_elements(merged_elements)
        
    def _merge_elements(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge existing and new story elements.
        
        Args:
            existing: Existing story elements
            new: New story elements
            
        Returns:
            Merged story elements
        """
        merged = {}
        
        # Merge style
        merged['style'] = new['style']  # Always use the latest style
        
        # Merge characters
        merged['characters'] = existing.get('characters', {})
        for char_name, char_info in new['characters'].items():
            if char_name in merged['characters']:
                # Update existing character
                merged['characters'][char_name].update(char_info)
            else:
                # Add new character
                merged['characters'][char_name] = char_info
                
        # Merge world
        merged['world'] = existing.get('world', {})
        merged['world'].update(new['world'])
        
        # Merge themes
        existing_themes = set(existing.get('themes', []))
        new_themes = set(new['themes'])
        merged['themes'] = list(existing_themes.union(new_themes))
        
        return merged 