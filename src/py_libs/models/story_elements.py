from typing import List, Dict
from pydantic import BaseModel, Field, ConfigDict

class StyleConfig(BaseModel):
    """Represents the style configuration of the story."""
    tone: str = Field(..., description="Overall tone of the story")
    pacing: str = Field(..., description="Description of the story's pacing")
    narrative_style: str = Field(..., description="Description of the narrative style")

class CharacterInfo(BaseModel):
    """Represents basic character information in story elements."""
    role: str = Field(..., description="Character's role in the story")
    traits: List[str] = Field(..., description="List of character traits")
    arc: str = Field(..., description="Description of character's story arc")

class WorldConfig(BaseModel):
    """Represents the world configuration of the story."""
    setting: str = Field(..., description="Description of the main setting")
    rules: List[str] = Field(..., description="List of world rules or phenomena")
    atmosphere: str = Field(..., description="Description of the world's atmosphere")

class StoryElements(BaseModel):
    """Represents the core elements of a story."""
    style: StyleConfig = Field(..., description="Style configuration")
    characters: Dict[str, CharacterInfo] = Field(..., description="Character information keyed by name")
    world: WorldConfig = Field(..., description="World configuration")
    themes: List[str] = Field(..., description="List of story themes") 