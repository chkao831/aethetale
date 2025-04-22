from pathlib import Path
import json
import yaml
from typing import Dict, Any, List
from openai import OpenAI
from .model_config import ModelConfig

class ConfigLoader:
    def __init__(self, story_path: Path, openai_client: OpenAI | None = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the config loader for a story.
        
        Args:
            story_path: Path to the story directory
            openai_client: OpenAI client instance to use, if None a new one will be created
            model: Model to use for generation
        """
        self.story_path = story_path
        self.shared_config_path = Path("config/shared")
        self.prompt_path = self.shared_config_path / "prompt.yaml"
        self.beats_path = story_path / "beats.yaml"  # Only beats are story-specific
        self.client = openai_client if openai_client is not None else OpenAI()
        self.model = model
        self.model_config = ModelConfig()
        
        # Ensure shared config directory exists
        if not self.shared_config_path.exists():
            raise FileNotFoundError(f"Shared config directory not found at {self.shared_config_path}")
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load the story configuration.
        
        Returns:
            Dictionary containing the story configuration
        """
        # Get model settings and chunk settings from ModelConfig
        model_settings = self.model_config.get_model_config()
        
        # Combine all settings
        config = {
            "model": self.model,
            "max_tokens": model_settings["max_tokens"],
            "temperature": model_settings["temperature"],
            "chunk_size": self.model_config.get_chunk_size(),
            "chunk_overlap": self.model_config.get_chunk_overlap(),
            "supported_languages": ["en", "zh"]
        }
        return config
            
    def load_prompts(self) -> Dict[str, Any]:
        """
        Load the prompt templates.
        
        Returns:
            Dictionary containing the prompt templates
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Shared prompt file not found at {self.prompt_path}")
            
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
            # Ensure all required prompts are present
            required_prompts = ["beat_expansion", "style_guidance", "character_extraction", "story_analysis"]
            for prompt in required_prompts:
                if prompt not in prompts:
                    prompts[prompt] = f"Default prompt for {prompt}: {{text}}"
            return prompts
            
    def load_beats(self) -> List[Dict[str, str]]:
        """
        Load the story beats.
        
        Returns:
            List of story beats
            
        Raises:
            FileNotFoundError: If the beats file doesn't exist
        """
        if not self.beats_path.exists():
            raise FileNotFoundError(f"Beats file not found at {self.beats_path}")
            
        with open(self.beats_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data["beats"]

    def analyze_beat(self, beat_description: str, story_context: str = None) -> Dict[str, Any]:
        """
        Analyze a beat description using LLM to determine character, context, and style.
        
        Args:
            beat_description: The beat description to analyze
            story_context: Optional context from the story
            
        Returns:
            Dictionary containing analyzed beat information
        """
        if not self.client:
            raise ValueError("OpenAI client not provided for beat analysis")
            
        # Load prompts
        prompts = self.load_prompts()
        
        # Prepare the analysis prompt
        analysis_prompt = prompts.get('beat_analysis', """
        Analyze the following narrative beat and determine:
        1. Which character is primarily involved
        2. What type of context this beat represents (e.g., character_introduction, world_building, etc.)
        3. Appropriate style settings (tone, point of view, tense)
        
        Beat: {beat}
        
        Story Context (if available):
        {context}
        
        Provide the analysis in JSON format with the following structure:
        {{
            "character": "character_name",
            "context": "context_type",
            "style": {{
                "tone": "tone_description",
                "pov": "point_of_view",
                "tense": "tense"
            }}
        }}
        """).format(beat=beat_description, context=story_context or "No context provided")
        
        # Get analysis from LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a narrative analysis assistant."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3
        )
        
        # Parse the response
        try:
            analysis = json.loads(response.choices[0].message.content)
            return {
                "name": beat_description,
                "position": 0,  # Will be set by the caller
                **analysis
            }
        except json.JSONDecodeError:
            # Fallback to basic analysis if JSON parsing fails
            return {
                "name": beat_description,
                "position": 0,
                "character": "main_character",
                "context": "general",
                "style": {
                    "tone": "neutral",
                    "pov": "third_person",
                    "tense": "past"
                }
            }
            
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default story configuration."""
        config = {
            "story_name": self.story_path.name,
            "model": {
                "name": self.model,
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
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
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
            """,
            "beat_analysis": """
            Analyze the following narrative beat and determine:
            1. Which character is primarily involved
            2. What type of context this beat represents (e.g., character_introduction, world_building, etc.)
            3. Appropriate style settings (tone, point of view, tense)
            
            Beat: {beat}
            
            Story Context (if available):
            {context}
            
            Provide the analysis in JSON format with the following structure:
            {{
                "character": "character_name",
                "context": "context_type",
                "style": {{
                    "tone": "tone_description",
                    "pov": "point_of_view",
                    "tense": "tense"
                }}
            }}
            """
        }
        
        with open(self.prompt_path, 'w', encoding='utf-8') as f:
            yaml.dump(prompts, f, allow_unicode=True)
            
        return prompts

    def _create_default_beats(self) -> List[str]:
        """Create default beats configuration."""
        beats = ["Initial story setup and character introduction"]
        
        beats_config = {"beats": beats}
        with open(self.beats_path, 'w', encoding='utf-8') as f:
            yaml.dump(beats_config, f, allow_unicode=True)
            
        return beats 