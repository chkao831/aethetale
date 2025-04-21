from typing import Dict, Any, Optional
from openai import OpenAI
from pathlib import Path
import json

class StoryGenerator:
    def __init__(self, story_path: Path, openai_client: OpenAI | None = None):
        """
        Initialize the story generator for a story.
        
        Args:
            story_path: Path to the story directory
            openai_client: OpenAI client instance to use, if None a new one will be created
        """
        self.story_path = story_path
        self.config_path = story_path / "config.json"
        self.client = openai_client if openai_client is not None else OpenAI()
        self._load_config()
        
    def _load_config(self):
        """Load model configuration."""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            self.model_config = config["model"]
            
    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: The prompt to generate from
            temperature: Optional temperature override
            
        Returns:
            Generated text
        """
        # Use provided temperature or default from config
        temp = temperature if temperature is not None else self.model_config["temperature"]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_config["name"],
                messages=[
                    {"role": "system", "content": "You are a creative writing assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=self.model_config["max_tokens"]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error generating text: {str(e)}")
            
    def expand_beat(self, beat: str, context: str, style: Dict[str, str]) -> str:
        """
        Expand a narrative beat into a scene.
        
        Args:
            beat: The narrative beat
            context: Relevant context
            style: Style configuration
            
        Returns:
            Expanded scene text
        """
        prompt = f"""
        Expand the following narrative beat into a detailed scene:
        
        Beat: {beat}
        
        Context from previous story:
        {context}
        
        Style guidelines:
        - Tone: {style['tone']}
        - Point of view: {style['pov']}
        - Tense: {style['tense']}
        
        Write a detailed scene that:
        1. Maintains consistency with the provided context
        2. Follows the style guidelines
        3. Develops the characters and their relationships
        4. Advances the plot naturally
        """
        
        return self.generate_text(prompt)
        
    def analyze_style(self, text: str) -> Dict[str, Any]:
        """
        Analyze the style of a text passage.
        
        Args:
            text: Text to analyze
            
        Returns:
            Style analysis results
        """
        prompt = f"""
        Analyze the following text and provide style guidance:
        
        Text: {text}
        
        Consider:
        - Tone and mood
        - Narrative voice
        - Character voice consistency
        - Pacing and structure
        
        Provide your analysis in JSON format with these fields:
        - tone: The overall tone of the passage
        - narrative_voice: Description of the narrative voice
        - character_consistency: Assessment of character voice consistency
        - pacing: Analysis of the pacing
        - strengths: List of stylistic strengths
        - suggestions: List of improvement suggestions
        """
        
        response = self.generate_text(prompt, temperature=0.3)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse style analysis",
                "raw_response": response
            } 