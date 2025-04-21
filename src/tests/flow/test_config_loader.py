import pytest
from pathlib import Path
import json
import yaml
from src.py_libs.flow.config_loader import ConfigLoader

@pytest.fixture
def temp_story_dir(tmp_path):
    """Create a temporary story directory for testing."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    return story_dir

def test_config_loader_initialization(temp_story_dir):
    """Test ConfigLoader initialization."""
    loader = ConfigLoader(temp_story_dir)
    assert loader.story_path == temp_story_dir
    assert loader.config_path == temp_story_dir / "config.json"
    assert loader.prompt_path == temp_story_dir / "prompt.yaml"

def test_load_config_with_existing_file(temp_story_dir):
    """Test loading config from existing file."""
    # Create test config
    test_config = {
        "story_name": "Test Story",
        "model": {"name": "test-model", "temperature": 0.7},
        "style": {"tone": "neutral"}
    }
    with open(temp_story_dir / "config.json", "w") as f:
        json.dump(test_config, f)
    
    loader = ConfigLoader(temp_story_dir)
    config = loader.load_config()
    assert config == test_config

def test_load_config_without_file(temp_story_dir):
    """Test loading config when file doesn't exist."""
    loader = ConfigLoader(temp_story_dir)
    config = loader.load_config()
    
    assert "story_name" in config
    assert config["story_name"] == "test_story"
    assert "model" in config
    assert "style" in config
    assert (temp_story_dir / "config.json").exists()

def test_load_prompts_with_existing_file(temp_story_dir):
    """Test loading prompts from existing file."""
    # Create test prompts
    test_prompts = {
        "beat_expansion": "Test prompt",
        "style_guidance": "Style test"
    }
    with open(temp_story_dir / "prompt.yaml", "w") as f:
        yaml.dump(test_prompts, f)
    
    loader = ConfigLoader(temp_story_dir)
    prompts = loader.load_prompts()
    assert prompts == test_prompts

def test_load_prompts_without_file(temp_story_dir):
    """Test loading prompts when file doesn't exist."""
    loader = ConfigLoader(temp_story_dir)
    prompts = loader.load_prompts()
    
    assert "beat_expansion" in prompts
    assert "style_guidance" in prompts
    assert (temp_story_dir / "prompt.yaml").exists() 