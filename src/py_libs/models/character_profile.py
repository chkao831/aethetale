from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class FamilyRelation(BaseModel):
    """Represents a family relationship between characters."""
    relation_type: str = Field(..., description="Type of family relationship (e.g., parent, sibling)")
    name: str = Field(..., description="Name of the related family member")

class CharacterProfile(BaseModel):
    """Represents a character's profile with rich attributes."""
    name: str = Field(..., description="Canonical name of the character")
    aliases: List[str] = Field(default_factory=list, description="List of alternative names or titles")
    role: str = Field(..., description="Narrative or mythic role in the story")
    occupation: str = Field("", description="Character's occupation or primary activity")
    personality_traits: List[str] = Field(default_factory=list, description="Core emotional/behavioral traits")
    goals: List[str] = Field(default_factory=list, description="Personal motivations and objectives")
    fears: List[str] = Field(default_factory=list, description="Deepest anxieties and concerns")
    lovers: List[str] = Field(default_factory=list, description="Names of romantic interests")
    friends: List[str] = Field(default_factory=list, description="Names of allies and friends")
    enemies: List[str] = Field(default_factory=list, description="Names of antagonists and enemies")
    family: List[FamilyRelation] = Field(default_factory=list, description="Family relationships")
    key_events: List[str] = Field(default_factory=list, description="Important scenes or events involving the character")
    style_embedding: Optional[List[float]] = Field(None, description="Vector embedding for character's style/voice")
    profile_text: Optional[str] = Field(None, description="Canonical summary of the character")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of profile creation")
    updated_at: datetime = Field(default_factory=datetime.now, description="Timestamp of last profile update")

    def __post_init__(self):
        """Validate character profile data after initialization."""
        if not self.name:
            raise ValueError("Character name cannot be empty")
        if not self.role:
            raise ValueError("Character role cannot be empty")
            
    def to_dict(self) -> dict:
        """Convert character profile to dictionary format."""
        return {
            "name": self.name,
            "aliases": self.aliases,
            "role": self.role,
            "occupation": self.occupation,
            "personality_traits": self.personality_traits,
            "goals": self.goals,
            "fears": self.fears,
            "lovers": self.lovers,
            "friends": self.friends,
            "enemies": self.enemies,
            "family": self.family,
            "key_events": self.key_events,
            "style_embedding": self.style_embedding,
            "profile_text": self.profile_text,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CharacterProfile':
        """Create character profile from dictionary data."""
        return cls(
            name=data["name"],
            aliases=data.get("aliases", []),
            role=data["role"],
            occupation=data.get("occupation", ""),
            personality_traits=data.get("personality_traits", []),
            goals=data.get("goals", []),
            fears=data.get("fears", []),
            lovers=data.get("lovers", []),
            friends=data.get("friends", []),
            enemies=data.get("enemies", []),
            family=data.get("family", []),
            key_events=data.get("key_events", []),
            style_embedding=data.get("style_embedding"),
            profile_text=data.get("profile_text"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        ) 