from typing import List, Dict, Any
from pathlib import Path
from src.py_libs.flow.config_loader import ConfigLoader
from src.py_libs.flow.retriever import ContextRetriever
from src.py_libs.flow.prompt_builder import PromptBuilder
from src.py_libs.flow.generator import StoryGenerator
from src.py_libs.flow.chapter_stitcher import ChapterStitcher
from openai import OpenAI

class FlowEngine:
    def __init__(self, story_path: Path, openai_client: OpenAI | None = None):
        """
        Initialize the flow engine for a story.
        
        Args:
            story_path: Path to the story directory
            openai_client: OpenAI client instance to use, if None a new one will be created
        """
        self.story_path = story_path
        self.config_loader = ConfigLoader(story_path)
        self.retriever = ContextRetriever(story_path)
        self.prompt_builder = PromptBuilder(story_path)
        self.generator = StoryGenerator(story_path, openai_client=openai_client)
        self.stitcher = ChapterStitcher(story_path)
        
    def generate_chapter(self, beats: List[str], chapter_number: int) -> Dict[str, Any]:
        """
        Generate a chapter from narrative beats.
        
        Args:
            beats: List of narrative beats
            chapter_number: Chapter number
            
        Returns:
            Dictionary containing the generated chapter and metadata
        """
        # Load configuration
        config = self.config_loader.load_config()
        style = config["style"]
        
        # Generate scenes for each beat
        scenes = []
        for i, beat in enumerate(beats):
            # Retrieve relevant context
            context = self.retriever.retrieve_context(beat)
            
            # Build prompt
            prompt = self.prompt_builder.build_beat_prompt(beat, context, style)
            
            # Generate scene
            scene_text = self.generator.expand_beat(beat, context, style)
            
            # Analyze style
            style_analysis = self.generator.analyze_style(scene_text)
            
            scenes.append({
                'text': scene_text,
                'position': i,
                'beat': beat,
                'style_analysis': style_analysis
            })
            
        # Stitch scenes into chapter
        chapter_text = self.stitcher.stitch_scenes(scenes)
        
        # Prepare metadata
        metadata = {
            'chapter_number': chapter_number,
            'beats': beats,
            'scenes': [{
                'position': scene['position'],
                'beat': scene['beat'],
                'style_analysis': scene['style_analysis']
            } for scene in scenes]
        }
        
        # Save chapter
        self.stitcher.save_chapter(chapter_text, chapter_number, metadata)
        
        return {
            'text': chapter_text,
            'metadata': metadata
        }
        
    def get_character_context(self, character_name: str) -> List[Dict[str, Any]]:
        """
        Get context about a character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            List of relevant context chunks
        """
        return self.retriever.get_character_context(character_name)
        
    def get_relationship_context(self, character1: str, character2: str) -> List[Dict[str, Any]]:
        """
        Get context about a relationship between two characters.
        
        Args:
            character1: First character name
            character2: Second character name
            
        Returns:
            List of relevant context chunks
        """
        return self.retriever.get_relationship_context(character1, character2)
        
    def analyze_style(self, text: str) -> Dict[str, Any]:
        """
        Analyze the style of a text passage.
        
        Args:
            text: Text to analyze
            
        Returns:
            Style analysis results
        """
        return self.generator.analyze_style(text)
        
    def list_chapters(self) -> List[int]:
        """
        Get a list of available chapters.
        
        Returns:
            List of chapter numbers
        """
        return self.stitcher.list_chapters()
        
    def load_chapter(self, chapter_number: int) -> Dict[str, Any]:
        """
        Load a chapter and its metadata.
        
        Args:
            chapter_number: Chapter number to load
            
        Returns:
            Dictionary with chapter text and metadata
        """
        return self.stitcher.load_chapter(chapter_number) 