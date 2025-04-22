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
            
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration for the default model.
        
        Returns:
            Dictionary containing model configuration
        """
        model_name = self.config["default_model"]
        return self.config["models"][model_name]
        
    def get_model_name(self) -> str:
        """
        Get the default model name.
        
        Returns:
            Model name string
        """
        return self.get_model_config()["name"]
        
    def get_temperature(self) -> float:
        """
        Get the temperature setting for the default model.
        
        Returns:
            Temperature value
        """
        return self.get_model_config()["temperature"]
        
    def get_max_tokens(self) -> int:
        """
        Get the max tokens setting for the default model.
        
        Returns:
            Max tokens value
        """
        return self.get_model_config()["max_tokens"]
        
    def get_chunk_size(self) -> int:
        """
        Get the chunk size setting.
        
        Returns:
            Chunk size value
        """
        return self.config.get("chunk_size", 1000)
        
    def get_chunk_overlap(self) -> int:
        """
        Get the chunk overlap setting.
        
        Returns:
            Chunk overlap value
        """
        return self.config.get("chunk_overlap", 200) 