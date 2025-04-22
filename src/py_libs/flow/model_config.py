from typing import Dict, Any
from pathlib import Path
import json

class ModelConfig:
    def __init__(self):
        """Initialize the model configuration manager."""
        self.config_path = Path("config/shared/model_config.json")
        self._load_config()
        
    def _load_config(self):
        """Load the model configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config file not found at {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
    def get_model_config(self, language: str = None) -> Dict[str, Any]:
        """
        Get the model configuration for a specific language or the default model.
        
        Args:
            language: Language code (e.g., 'zh', 'en') or None for default
            
        Returns:
            Dictionary containing model configuration
        """
        if language and language in self.config["language_models"]:
            model_name = self.config["language_models"][language]
        else:
            model_name = self.config["default_model"]
            
        return self.config["models"][model_name]
        
    def get_model_name(self, language: str = None) -> str:
        """
        Get the model name for a specific language or the default model.
        
        Args:
            language: Language code (e.g., 'zh', 'en') or None for default
            
        Returns:
            Model name string
        """
        return self.get_model_config(language)["name"]
        
    def get_temperature(self, language: str = None) -> float:
        """
        Get the temperature setting for a specific language or the default model.
        
        Args:
            language: Language code (e.g., 'zh', 'en') or None for default
            
        Returns:
            Temperature value
        """
        return self.get_model_config(language)["temperature"]
        
    def get_max_tokens(self, language: str = None) -> int:
        """
        Get the max tokens setting for a specific language or the default model.
        
        Args:
            language: Language code (e.g., 'zh', 'en') or None for default
            
        Returns:
            Max tokens value
        """
        return self.get_model_config(language)["max_tokens"] 