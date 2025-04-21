import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch
from src.py_libs.flow.generator import StoryGenerator

@pytest.fixture
def temp_story_dir(tmp_path):
    """Create a temporary story directory with test config."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    
    # Create shared config directory and config
    shared_config_dir = tmp_path / "config" / "shared"
    shared_config_dir.mkdir(parents=True)
    
    config = {
        "model": {
            "name": "gpt-4-turbo-preview",
            "temperature": 0.7,
            "max_tokens": 2000
        }
    }
    
    with open(shared_config_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    return story_dir

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="Generated text"))
    ]
    return client

@patch('src.py_libs.flow.generator.OpenAI')
def test_generator_initialization(mock_openai_class, temp_story_dir, mock_openai_client):
    """Test StoryGenerator initialization."""
    # Set up mock for default client
    mock_openai_class.return_value = Mock()
    mock_openai_class.return_value.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="Generated text"))
    ]
    
    # Test with provided client
    generator = StoryGenerator(temp_story_dir, mock_openai_client, language="en")
    assert generator.story_path == temp_story_dir
    assert generator.client == mock_openai_client
    assert generator.language == "en"
    
    # Test with default client
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        generator = StoryGenerator(temp_story_dir, language="zh")
        assert generator.story_path == temp_story_dir
        assert isinstance(generator.client, Mock)
        assert generator.language == "zh"
        mock_openai_class.assert_called_once()

def test_generate_text(temp_story_dir, mock_openai_client):
    """Test text generation."""
    generator = StoryGenerator(temp_story_dir, mock_openai_client, language="en")
    prompt = "Test prompt"
    
    text = generator.generate_text(prompt)
    assert text == "Generated text"
    
    # Verify API call
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4-turbo-preview"
    assert call_args["temperature"] == 0.7
    assert call_args["max_tokens"] == 2000

def test_expand_beat(temp_story_dir, mock_openai_client):
    """Test beat expansion."""
    generator = StoryGenerator(temp_story_dir, mock_openai_client, language="en")
    beat = "Test beat"
    context = "Test context"
    style = {
        "tone": "neutral",
        "pov": "third_person",
        "tense": "past"
    }
    
    text = generator.expand_beat(beat, context, style)
    assert text == "Generated text"
    
    # Verify API call
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert beat in call_args["messages"][1]["content"]
    assert context in call_args["messages"][1]["content"]
    assert "language" in call_args["messages"][1]["content"]
    assert "en" in call_args["messages"][1]["content"]

def test_analyze_style(temp_story_dir, mock_openai_client):
    """Test style analysis."""
    generator = StoryGenerator(temp_story_dir, mock_openai_client, language="en")
    text = "Test text for style analysis"
    
    # Test successful JSON parsing
    mock_openai_client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"tone": "neutral", "narrative_voice": "descriptive"}'))
    ]
    
    analysis = generator.analyze_style(text)
    assert isinstance(analysis, dict)
    assert "tone" in analysis
    assert "narrative_voice" in analysis
    
    # Test failed JSON parsing
    mock_openai_client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="Invalid JSON"))
    ]
    
    analysis = generator.analyze_style(text)
    assert "error" in analysis
    assert "raw_response" in analysis 