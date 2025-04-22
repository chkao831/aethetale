from typing import List, Dict, Any
from pathlib import Path
import yaml

class PromptBuilder:
    def __init__(self, story_path: Path, language: str = "en"):
        """
        Initialize the prompt builder.
        
        Args:
            story_path: Path to the story directory
            language: Language to use for prompts (default: "en")
        """
        self.story_path = story_path
        self.shared_config_path = Path("config/shared")
        self.prompt_path = self.shared_config_path / "prompt.yaml"
        self.language = language
        self._load_prompts()
        
    def _load_prompts(self):
        """Load prompt templates from the shared configuration."""
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
            # Ensure all required prompts are present
            required_prompts = {
                "beat_expansion": """
                Write a continuous narrative in {language} that seamlessly continues the story, incorporating the following story beats:
                {beats}
                
                Previous story context:
                {context}
                
                {character_contexts}
                
                Style guidelines:
                - Tone: {tone}
                - Point of View: {pov}
                - Tense: {tense}
                
                Write a detailed scene that:
                1. Maintains consistency with the provided context
                2. Follows the style guidelines
                3. Develops the characters and their relationships based on their established profiles
                4. Advances the plot naturally
                """,
                "style_guidance": "Analyze style: {text}",
                "character_extraction": "Extract character details: {text}",
                "story_analysis": "Analyze story: {text}"
            }
            for prompt_name, default_prompt in required_prompts.items():
                if prompt_name not in self.prompts:
                    self.prompts[prompt_name] = default_prompt
            
    def build_beat_prompt(self, beats: str, context: str, style: dict, character_contexts: dict = None) -> str:
        """Build a prompt for generating continuous narrative from story beats."""
        # Add character context section if available
        character_context_section = ""
        if character_contexts:
            character_context_section = "\nCharacter Contexts:\n"
            for character_name, contexts in character_contexts.items():
                character_context_section += f"\n{character_name}:\n"
                for i, context_chunk in enumerate(contexts):
                    character_context_section += f"Context {i+1}:\n{context_chunk['text']}\n"
        
        # Format the base prompt
        base_prompt = self.prompts['beat_expansion'].format(
            beats=beats,
            context=context,
            tone=style.get('tone', 'neutral'),
            pov=style.get('pov', 'third person'),
            tense=style.get('tense', 'past'),
            language=self.language,
            character_contexts=character_context_section
        )
        
        # Add language-specific instructions
        if self.language == "zh":
            base_prompt += """
            
            重要提示：这是一个中文故事，你必须：
            1. 完全使用中文写作
            2. 适当使用中文成语和表达方式
            3. 保持中文文学风格和传统
            4. 使用中文标点符号和格式
            5. 保持所有角色名称和术语的中文形式
            6. 遵循中文叙事传统和故事讲述模式
            7. 使用中文特有的文学手法和修辞技巧
            8. 保持与输入文本相同的中文文学水准
            
            请记住：整个回答必须使用中文，包括所有叙事元素、描述和对话。
            """
        
        return base_prompt
        
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
        
        return self.prompts["character_extraction"].format(
            text=context_text
        )
        
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