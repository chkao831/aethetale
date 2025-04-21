from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from .beat import Beat

class Chapter(BaseModel):
    """Represents a chapter in the story."""
    id: str = Field(..., description="Unique identifier for the chapter")
    title: str = Field(..., description="Title of the chapter")
    content: str = Field(..., description="The actual content of the chapter")
    beats: List[str] = Field(..., description="List of beat IDs that make up this chapter")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the chapter")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of chapter creation")
    updated_at: datetime = Field(default_factory=datetime.now, description="Timestamp of last chapter update")
