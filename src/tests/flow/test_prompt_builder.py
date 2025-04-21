import pytest
from pathlib import Path
import yaml
from src.py_libs.flow.prompt_builder import PromptBuilder

@pytest.fixture
def temp_story_dir(tmp_path):
    """Create a temporary story directory with test prompts."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    # Create shared config directory and prompts
    shared_config_dir = tmp_path / "config" / "shared"
    shared_config_dir.mkdir(parents=True)
    
    # Create test prompts with language support
    prompts = {
        "beat_expansion": """
        Write a continuous narrative in {language} that seamlessly continues the story, incorporating the following story beats:
        {beats}
        
        Previous story context:
        {context}
        
        Style guidelines:
        - Tone: {tone}
        - Point of View: {pov}
        - Tense: {tense}
        """,
        "style_guidance": "Analyze style: {text}",
        "character_extraction": "Extract character details: {text}",
        "story_analysis": "Analyze story: {text}"
    }
    
    with open(shared_config_dir / "prompt.yaml", "w", encoding='utf-8') as f:
        yaml.dump(prompts, f)
    
    return story_dir

def test_prompt_builder_initialization(temp_story_dir):
    """Test PromptBuilder initialization."""
    builder = PromptBuilder(temp_story_dir, "en")  # Test English
    assert builder.story_path == temp_story_dir
    assert builder.prompt_path == Path("config/shared/prompt.yaml")
    assert "beat_expansion" in builder.prompts
    assert "style_guidance" in builder.prompts
    
    # Test Chinese initialization
    builder_zh = PromptBuilder(temp_story_dir, "zh")
    assert builder_zh.language == "zh"

def test_build_beat_prompt(temp_story_dir):
    """Test building beat expansion prompt."""
    # Test English
    builder_en = PromptBuilder(temp_story_dir, "en")
    beat = "Test beat"
    context = "Context 1"
    style = {
        "tone": "neutral",
        "pov": "third person",
        "tense": "past"
    }
    
    prompt_en = builder_en.build_beat_prompt(beat, context, style)
    assert "write a continuous narrative that seamlessly continues the story" in prompt_en.lower()
    assert "test beat" in prompt_en.lower()
    assert "context 1" in prompt_en.lower()
    assert "tone: neutral" in prompt_en.lower()
    assert "point of view: third person" in prompt_en.lower()
    assert "tense: past" in prompt_en.lower()
    assert "language: en" in prompt_en.lower()
    
    # Test Chinese
    builder_zh = PromptBuilder(temp_story_dir, "zh")
    prompt_zh = builder_zh.build_beat_prompt(beat, context, style)
    assert "write a continuous narrative that seamlessly continues the story" in prompt_zh.lower()
    assert "language: zh" in prompt_zh.lower()

def test_build_style_prompt(temp_story_dir):
    """Test building style analysis prompt."""
    builder = PromptBuilder(temp_story_dir, "en")
    text = "Test text for style analysis"
    
    prompt = builder.build_style_prompt(text)
    assert text in prompt

def test_build_character_prompt(temp_story_dir):
    """Test building character development prompt."""
    builder = PromptBuilder(temp_story_dir, "en")
    character = "Alice"
    context = [{"text": "Character context 1"}]
    
    prompt = builder.build_character_prompt(character, context)
    assert "Character context 1" in prompt

def test_build_relationship_prompt(temp_story_dir):
    """Test building relationship development prompt."""
    builder = PromptBuilder(temp_story_dir, "en")
    character1 = "Alice"
    character2 = "Bob"
    context = [{"text": "Relationship context 1"}]
    
    prompt = builder.build_relationship_prompt(character1, character2, context)
    assert character1 in prompt
    assert character2 in prompt
    assert "Relationship context 1" in prompt 