import pytest
from pathlib import Path
import yaml
from src.py_libs.flow.prompt_builder import PromptBuilder

@pytest.fixture
def temp_story_dir(tmp_path):
    """Create a temporary story directory with test prompts."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    # Create test prompts
    prompts = {
        "beat_expansion": "Expand beat: {beat}\nContext: {context}\nStyle: {tone}, {pov}, {tense}",
        "style_guidance": "Analyze style: {text}"
    }
    
    with open(story_dir / "prompt.yaml", "w") as f:
        yaml.dump(prompts, f)
    
    return story_dir

def test_prompt_builder_initialization(temp_story_dir):
    """Test PromptBuilder initialization."""
    builder = PromptBuilder(temp_story_dir)
    assert builder.story_path == temp_story_dir
    assert builder.prompt_path == temp_story_dir / "prompt.yaml"
    assert "beat_expansion" in builder.prompts
    assert "style_guidance" in builder.prompts

def test_build_beat_prompt(temp_story_dir):
    """Test building beat expansion prompt."""
    builder = PromptBuilder(temp_story_dir)
    beat = "Test beat"
    context = [{"text": "Context 1", "similarity_score": 0.8}]
    style = {
        "tone": "neutral",
        "pov": "third_person",
        "tense": "past"
    }
    
    prompt = builder.build_beat_prompt(beat, context, style)
    assert beat in prompt
    assert "Context 1" in prompt
    assert "neutral" in prompt
    assert "third_person" in prompt
    assert "past" in prompt

def test_build_style_prompt(temp_story_dir):
    """Test building style analysis prompt."""
    builder = PromptBuilder(temp_story_dir)
    text = "Test text for style analysis"
    
    prompt = builder.build_style_prompt(text)
    assert text in prompt

def test_build_character_prompt(temp_story_dir):
    """Test building character development prompt."""
    builder = PromptBuilder(temp_story_dir)
    character = "Alice"
    context = [{"text": "Character context 1"}]
    
    prompt = builder.build_character_prompt(character, context)
    assert character in prompt
    assert "Character context 1" in prompt

def test_build_relationship_prompt(temp_story_dir):
    """Test building relationship development prompt."""
    builder = PromptBuilder(temp_story_dir)
    character1 = "Alice"
    character2 = "Bob"
    context = [{"text": "Relationship context 1"}]
    
    prompt = builder.build_relationship_prompt(character1, character2, context)
    assert character1 in prompt
    assert character2 in prompt
    assert "Relationship context 1" in prompt 