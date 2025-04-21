from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Beat:
    """Represents a narrative beat in the story."""
    id: str
    content: str
    priority_fields: List[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    dependencies: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate beat data after initialization."""
        if not self.id:
            raise ValueError("Beat ID cannot be empty")
        if not self.content:
            raise ValueError("Beat content cannot be empty")
        if not self.priority_fields:
            raise ValueError("Priority fields cannot be empty")
            
    def to_dict(self) -> dict:
        """Convert beat to dictionary format."""
        return {
            "id": self.id,
            "content": self.content,
            "priority_fields": self.priority_fields,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "dependencies": self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Beat':
        """Create beat from dictionary data."""
        return cls(
            id=data["id"],
            content=data["content"],
            priority_fields=data["priority_fields"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            dependencies=data.get("dependencies")
        )
