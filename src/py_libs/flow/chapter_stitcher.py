from typing import List, Dict, Any
from pathlib import Path
import json

class ChapterStitcher:
    def __init__(self, story_path: Path):
        """
        Initialize the chapter stitcher for a story.
        
        Args:
            story_path: Path to the story directory
        """
        self.story_path = story_path
        
    def stitch_scenes(self, scenes: List[Dict[str, Any]]) -> str:
        """
        Stitch together scenes into a cohesive chapter.
        
        Args:
            scenes: List of scene dictionaries with text and metadata
            
        Returns:
            Stitched chapter text
        """
        # Sort scenes by their position in the narrative
        sorted_scenes = sorted(scenes, key=lambda x: x.get('position', 0))
        
        # Combine scene texts with transitions
        chapter_parts = []
        for i, scene in enumerate(sorted_scenes):
            text = scene['text']
            
            # Add transition if not the first scene
            if i > 0:
                transition = self._generate_transition(
                    sorted_scenes[i-1]['text'],
                    text
                )
                chapter_parts.append(transition)
                
            chapter_parts.append(text)
            
        return "\n\n".join(chapter_parts)
        
    def _generate_transition(self, previous_text: str, next_text: str) -> str:
        """
        Generate a transition between two scenes.
        
        Args:
            previous_text: Text of the previous scene
            next_text: Text of the next scene
            
        Returns:
            Transition text
        """
        # Simple transition for now - can be enhanced with LLM
        return "---"
        
    def save_chapter(self, chapter_text: str, chapter_number: int, metadata: Dict[str, Any]):
        """
        Save a chapter to disk.
        
        Args:
            chapter_text: The chapter text
            chapter_number: Chapter number
            metadata: Chapter metadata
        """
        # Create chapters directory if it doesn't exist
        chapters_dir = self.story_path / "chapters"
        chapters_dir.mkdir(exist_ok=True)
        
        # Save chapter text
        chapter_file = chapters_dir / f"chapter_{chapter_number}.md"
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
            
        # Save metadata
        metadata_file = chapters_dir / f"chapter_{chapter_number}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
    def load_chapter(self, chapter_number: int) -> Dict[str, Any]:
        """
        Load a chapter and its metadata.
        
        Args:
            chapter_number: Chapter number to load
            
        Returns:
            Dictionary with chapter text and metadata
        """
        chapters_dir = self.story_path / "chapters"
        
        # Load chapter text
        chapter_file = chapters_dir / f"chapter_{chapter_number}.md"
        with open(chapter_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Load metadata
        metadata_file = chapters_dir / f"chapter_{chapter_number}_metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        return {
            'text': text,
            'metadata': metadata
        }
        
    def list_chapters(self) -> List[int]:
        """
        Get a list of available chapter numbers.
        
        Returns:
            List of chapter numbers
        """
        chapters_dir = self.story_path / "chapters"
        if not chapters_dir.exists():
            return []
            
        chapters = []
        for file in chapters_dir.glob("chapter_*.md"):
            try:
                chapter_num = int(file.stem.split('_')[1])
                chapters.append(chapter_num)
            except (ValueError, IndexError):
                continue
                
        return sorted(chapters) 