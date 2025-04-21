from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

@dataclass
class PromptTemplate:
    """Represents a prompt template with variables."""
    template: str
    variables: Dict[str, str]
    metadata: Dict[str, str]
    
    def __post_init__(self):
        """Validate prompt template data."""
        if not self.template:
            raise ValueError("Template cannot be empty")
            
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
            
    @classmethod
    def from_file(cls, file_path: Path) -> 'PromptTemplate':
        """Load prompt template from file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read().strip()
            
        return cls(
            template=template,
            variables={},
            metadata={"source": str(file_path)}
        )
        
    def save_to_file(self, file_path: Path) -> None:
        """Save prompt template to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.template)
            
    def add_variable(self, name: str, value: str) -> None:
        """Add a variable to the template."""
        self.variables[name] = value
        
    def remove_variable(self, name: str) -> None:
        """Remove a variable from the template."""
        if name in self.variables:
            del self.variables[name]
