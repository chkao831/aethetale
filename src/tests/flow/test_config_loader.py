import pytest
from pathlib import Path
import yaml
import json
from src.py_libs.flow.config_loader import ConfigLoader

@pytest.fixture
def temp_story_dir(tmp_path):
    """Create a temporary story directory with test configurations."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    # Create shared config directory
    shared_config_dir = tmp_path / "config" / "shared"
    shared_config_dir.mkdir(parents=True)
    
    # Create shared config.json
    config = {
        "model": "gpt-4-turbo-preview",
        "temperature": 0.7,
        "max_tokens": 1000,
        "supported_languages": ["en", "zh"]
    }
    with open(shared_config_dir / "config.json", "w", encoding='utf-8') as f:
        json.dump(config, f)
    
    # Create shared prompt.yaml
    prompts = {
        "beat_expansion": "Expand beat in {language}: {beat}\nContext: {context}\nStyle: {tone}, {pov}, {tense}",
        "style_guidance": "Analyze style: {text}",
        "character_extraction": "Extract character details: {text}",
        "story_analysis": "Analyze story: {text}",
        "beat_analysis": "Analyze beat: {beat}\nContext: {context}"
    }
    with open(shared_config_dir / "prompt.yaml", "w", encoding='utf-8') as f:
        yaml.dump(prompts, f)
    
    # Create story-specific beats.yaml
    beats = {
        "beats": [
            {"id": "beat1", "text": "First beat"},
            {"id": "beat2", "text": "Second beat"}
        ]
    }
    with open(story_dir / "beats.yaml", "w", encoding='utf-8') as f:
        yaml.dump(beats, f)
    
    return story_dir

def test_config_loader_initialization(temp_story_dir):
    """Test ConfigLoader initialization."""
    loader = ConfigLoader(temp_story_dir)
    assert loader.story_path == temp_story_dir
    assert loader.shared_config_path == Path("config/shared")

def test_load_config(temp_story_dir):
    """Test loading configuration."""
    loader = ConfigLoader(temp_story_dir)
    config = loader.load_config()
    
    assert config["model"] == "gpt-4-turbo-preview"
    assert config["temperature"] == 0.7
    assert config["max_tokens"] == 1000
    assert "en" in config["supported_languages"]
    assert "zh" in config["supported_languages"]

def test_load_prompts(temp_story_dir):
    """Test loading prompts."""
    loader = ConfigLoader(temp_story_dir)
    prompts = loader.load_prompts()
    
    assert "beat_expansion" in prompts
    assert "{language}" in prompts["beat_expansion"]
    assert "style_guidance" in prompts
    assert "character_extraction" in prompts
    assert "story_analysis" in prompts

def test_load_beats(temp_story_dir):
    """Test loading beats."""
    loader = ConfigLoader(temp_story_dir)
    beats = loader.load_beats()
    
    assert len(beats) == 2
    assert beats[0]["id"] == "beat1"
    assert beats[1]["id"] == "beat2"

def test_missing_shared_config(tmp_path):
    """Test handling of missing shared configuration."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    # Create shared config directory but no files
    shared_config_dir = tmp_path / "config" / "shared"
    shared_config_dir.mkdir(parents=True)
    
    # Change working directory to tmp_path
    with pytest.MonkeyPatch().context() as mp:
        mp.chdir(tmp_path)
        loader = ConfigLoader(story_dir)
        
        # Should raise FileNotFoundError when trying to load missing files
        with pytest.raises(FileNotFoundError):
            loader.load_config()
            
        with pytest.raises(FileNotFoundError):
            loader.load_prompts()

def test_missing_beats(tmp_path):
    """Test handling of missing beats file."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    loader = ConfigLoader(story_dir)
    
    with pytest.raises(FileNotFoundError):
        loader.load_beats()

def test_analyze_beat(temp_story_dir, mocker):
    """Test beat analysis functionality."""
    # Mock OpenAI client response
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock()]
    mock_response.choices[0].message.content = json.dumps({
        "character": "test_character",
        "context": "test_context",
        "style": {
            "tone": "test_tone",
            "pov": "test_pov",
            "tense": "test_tense"
        }
    })
    
    # Mock OpenAI client
    mock_client = mocker.Mock()
    mock_client.chat.completions.create.return_value = mock_response
    
    # Initialize loader with mock client
    loader = ConfigLoader(temp_story_dir, mock_client)
    
    # Test successful analysis
    result = loader.analyze_beat("Test beat", "Test context")
    assert result["character"] == "test_character"
    assert result["context"] == "test_context"
    assert result["style"]["tone"] == "test_tone"
    
    # Test missing client
    loader_no_client = ConfigLoader(temp_story_dir)
    with pytest.raises(ValueError):
        loader_no_client.analyze_beat("Test beat")
        
    # Test JSON parsing error
    mock_response.choices[0].message.content = "invalid json"
    result = loader.analyze_beat("Test beat")
    assert result["character"] == "main_character"  # Default fallback
    assert result["style"]["tone"] == "neutral"  # Default fallback 