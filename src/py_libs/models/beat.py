from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class Beat(BaseModel):
    """Represents a narrative beat in the story."""
    id: str = Field(..., description="Unique identifier for the beat")
    content: str = Field(..., description="The actual content/description of the beat")
    priority_fields: List[str] = Field(..., description="Fields to prioritize during retrieval")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the beat")
    dependencies: Optional[List[str]] = Field(None, description="IDs of beats that this beat depends on")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of beat creation")
    updated_at: datetime = Field(default_factory=datetime.now, description="Timestamp of last beat update")
