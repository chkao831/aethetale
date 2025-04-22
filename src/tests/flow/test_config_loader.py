import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch
from src.py_libs.flow.config_loader import ConfigLoader
import yaml

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=json.dumps({
        "character": "test_character",
        "context": "test_context",
        "style": {
            "tone": "test_tone",
            "pov": "test_pov",
            "tense": "test_tense"
        }
    })))]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

@pytest.fixture
def temp_story_dir(tmp_path):
    """Create a temporary story directory with test config."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    # Create shared config directory and config
    shared_config_path = tmp_path / "config" / "shared"
    shared_config_path.mkdir(parents=True)
    
    # Create config.json
    config = {
        "model": "gpt-3.5-turbo",
        "max_tokens": 1000,
        "temperature": 0.7,
        "supported_languages": ["en", "zh"]
    }
    
    with open(shared_config_path / "config.json", "w") as f:
        json.dump(config, f)
    
    # Create beats.yaml
    beats = {
        "beats": [
            {"id": "beat1", "description": "First beat"},
            {"id": "beat2", "description": "Second beat"}
        ]
    }
    
    with open(story_dir / "beats.yaml", "w") as f:
        yaml.dump(beats, f)
    
    return story_dir

@patch('src.py_libs.flow.config_loader.OpenAI')
def test_load_config(mock_openai_class, temp_story_dir, mock_openai_client):
    """Test loading configuration."""
    mock_openai_class.return_value = mock_openai_client
    loader = ConfigLoader(temp_story_dir)
    config = loader.load_config()
    assert isinstance(config, dict)
    assert "model" in config
    assert "max_tokens" in config
    assert "temperature" in config
    assert "supported_languages" in config

@patch('src.py_libs.flow.config_loader.OpenAI')
def test_config_loader_initialization(mock_openai_class, temp_story_dir, mock_openai_client):
    """Test ConfigLoader initialization."""
    mock_openai_class.return_value = mock_openai_client
    loader = ConfigLoader(temp_story_dir)
    assert loader.story_path == temp_story_dir
    assert loader.shared_config_path == Path("config/shared")

@patch('src.py_libs.flow.config_loader.OpenAI')
def test_load_prompts(mock_openai_class, temp_story_dir, mock_openai_client):
    """Test loading prompts."""
    mock_openai_class.return_value = mock_openai_client
    loader = ConfigLoader(temp_story_dir)
    prompts = loader.load_prompts()
    
    assert "beat_expansion" in prompts
    assert "{language}" in prompts["beat_expansion"]
    assert "style_guidance" in prompts
    assert "character_extraction" in prompts
    assert "story_analysis" in prompts

@patch('src.py_libs.flow.config_loader.OpenAI')
def test_load_beats(mock_openai_class, temp_story_dir, mock_openai_client):
    """Test loading beats."""
    mock_openai_class.return_value = mock_openai_client
    loader = ConfigLoader(temp_story_dir)
    beats = loader.load_beats()
    
    assert len(beats) == 2
    assert beats[0]["id"] == "beat1"
    assert beats[1]["id"] == "beat2"

@patch('src.py_libs.flow.config_loader.OpenAI')
def test_missing_shared_config(mock_openai_class, tmp_path, mock_openai_client):
    """Test handling of missing shared configuration."""
    mock_openai_class.return_value = mock_openai_client
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

@patch('src.py_libs.flow.config_loader.OpenAI')
def test_missing_beats(mock_openai_class, tmp_path, mock_openai_client):
    """Test handling of missing beats file."""
    mock_openai_class.return_value = mock_openai_client
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    loader = ConfigLoader(story_dir)
    
    with pytest.raises(FileNotFoundError):
        loader.load_beats()

@patch('src.py_libs.flow.config_loader.OpenAI')
def test_analyze_beat(mock_openai_class, temp_story_dir, mock_openai_client):
    """Test beat analysis functionality."""
    mock_openai_class.return_value = mock_openai_client
    
    # Initialize loader with mock client
    loader = ConfigLoader(temp_story_dir)
    
    # Test successful analysis
    result = loader.analyze_beat("Test beat", "Test context")
    assert result["character"] == "test_character"
    assert result["context"] == "test_context"
    assert result["style"]["tone"] == "test_tone"
    
    # Test missing client
    with patch('src.py_libs.flow.config_loader.OpenAI', return_value=None):
        loader_no_client = ConfigLoader(temp_story_dir)
        with pytest.raises(ValueError):
            loader_no_client.analyze_beat("Test beat")
            
    # Test JSON parsing error
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="invalid json"))]
    mock_openai_client.chat.completions.create.return_value = mock_response
    result = loader.analyze_beat("Test beat")
    assert result["character"] == "main_character"  # Default fallback
    assert result["style"]["tone"] == "neutral"  # Default fallback 