from dataclasses import dataclass
from typing import List
from datetime import datetime
from .beat import Beat

@dataclass
class Chapter:
    """Represents a chapter in the story."""
    id: str
    title: str
    content: str
    beats: List[Beat]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        """Validate chapter data after initialization."""
        if not self.id:
            raise ValueError("Chapter ID cannot be empty")
        if not self.title:
            raise ValueError("Chapter title cannot be empty")
        if not self.content:
            raise ValueError("Chapter content cannot be empty")
            
    def to_dict(self) -> dict:
        """Convert chapter to dictionary format."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "beats": [beat.to_dict() for beat in self.beats],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Chapter':
        """Create chapter from dictionary data."""
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            beats=[Beat.from_dict(beat_data) for beat_data in data["beats"]],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
