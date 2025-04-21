from typing import Dict, Any, Optional
from pathlib import Path
import json
import shutil
from datetime import datetime

class VersionManager:
    def __init__(self, story_path: Path):
        """
        Initialize the version manager for a story.
        
        Args:
            story_path: Path to the story directory
        """
        self.story_path = story_path
        self.registry_path = story_path / "index_registry.json"
        self.versions_path = story_path / "versions"
        self.versions_path.mkdir(exist_ok=True)
        
        if not self.registry_path.exists():
            self._init_registry()
            
    def _init_registry(self):
        """Initialize the version registry file."""
        registry = {
            "current_version": None,
            "versions": []
        }
        self._save_registry(registry)
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load the version registry."""
        with open(self.registry_path, 'r') as f:
            return json.load(f)
            
    def _save_registry(self, registry: Dict[str, Any]):
        """Save the version registry."""
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
            
    def create_version(self, description: str) -> str:
        """
        Create a new version of the index.
        
        Args:
            description: Description of the version
            
        Returns:
            Version ID
        """
        registry = self._load_registry()
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Copy current index to new version
        version_path = self.versions_path / version_id
        version_path.mkdir()
        
        if registry["current_version"]:
            current_path = self.versions_path / registry["current_version"]
            for file in current_path.glob("*"):
                if file.is_dir():
                    shutil.copytree(file, version_path / file.name)
                else:
                    shutil.copy2(file, version_path / file.name)
                
        # Update registry
        registry["versions"].append({
            "id": version_id,
            "description": description,
            "created_at": datetime.now().isoformat()
        })
        registry["current_version"] = version_id
        self._save_registry(registry)
        
        return version_id
        
    def revert_to_version(self, version_id: str) -> bool:
        """
        Revert to a previous version.
        
        Args:
            version_id: ID of the version to revert to
            
        Returns:
            True if successful, False otherwise
        """
        registry = self._load_registry()
        
        # Check if version exists
        version_exists = any(v["id"] == version_id for v in registry["versions"])
        if not version_exists:
            return False
            
        # Copy version files to current
        version_path = self.versions_path / version_id
        for file in version_path.glob("*"):
            if file.is_dir():
                shutil.copytree(file, self.story_path / file.name, dirs_exist_ok=True)
            else:
                shutil.copy2(file, self.story_path / file.name)
            
        # Update registry
        registry["current_version"] = version_id
        self._save_registry(registry)
        
        return True
        
    def get_current_version(self) -> Optional[str]:
        """Get the current version ID."""
        registry = self._load_registry()
        return registry["current_version"]
        
    def list_versions(self) -> Dict[str, Any]:
        """Get a list of all versions."""
        registry = self._load_registry()
        return registry["versions"] 