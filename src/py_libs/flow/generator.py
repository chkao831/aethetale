from typing import Dict, Any, Optional
from openai import OpenAI
from pathlib import Path
import json

class StoryGenerator:
    def __init__(self, story_path: Path, openai_client: OpenAI | None = None, language: str = "en"):
        """
        Initialize the story generator.
        
        Args:
            story_path: Path to the story directory
            openai_client: OpenAI client instance (optional)
            language: Language to use for generation (default: "en")
        """
        self.story_path = story_path
        self.shared_config_path = Path("config/shared")
        self.client = openai_client if openai_client is not None else OpenAI()
        self.language = language
        self._load_config()
        
    def _load_config(self):
        """Load configuration from shared config directory."""
        with open(self.shared_config_path / "config.json", 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            # Ensure model is set to gpt-4-turbo-preview
            self.config["model"] = "gpt-4-turbo-preview"
            # Set a reasonable default for max_tokens if not specified
            if "max_tokens" not in self.config:
                self.config["max_tokens"] = 2000  # Default to 2000 tokens for full chapters
            
    def generate_text(self, prompt: str) -> str:
        """Generate text based on a prompt."""
        response = self.client.chat.completions.create(
            model=self.config.get("model", "gpt-4-turbo-preview"),
            messages=[
                {"role": "system", "content": f"You are a creative writing assistant. Generate text in {self.language} language."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 100)
        )
        return response.choices[0].message.content.strip()
            
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
        Expand the following narrative beat into a detailed scene in {self.language} language:
        
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
        
        response = self.generate_text(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse style analysis",
                "raw_response": response
            } 