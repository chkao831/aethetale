import pytest
from pathlib import Path
import json
from src.py_libs.flow.chapter_stitcher import ChapterStitcher

@pytest.fixture
def temp_story_dir(tmp_path):
    """Create a temporary story directory."""
    story_dir = tmp_path / "test_story"
    story_dir.mkdir()
    return story_dir

def test_chapter_stitcher_initialization(temp_story_dir):
    """Test ChapterStitcher initialization."""
    stitcher = ChapterStitcher(temp_story_dir)
    assert stitcher.story_path == temp_story_dir

def test_stitch_scenes(temp_story_dir):
    """Test scene stitching."""
    stitcher = ChapterStitcher(temp_story_dir)
    scenes = [
        {
            "text": "Scene 1",
            "position": 0,
            "beat": "Beat 1"
        },
        {
            "text": "Scene 2",
            "position": 1,
            "beat": "Beat 2"
        }
    ]
    
    chapter_text = stitcher.stitch_scenes(scenes)
    assert "Scene 1" in chapter_text
    assert "Scene 2" in chapter_text
    assert "---" in chapter_text  # Transition marker

def test_save_and_load_chapter(temp_story_dir):
    """Test saving and loading a chapter."""
    stitcher = ChapterStitcher(temp_story_dir)
    chapter_text = "Test chapter text"
    chapter_number = 1
    metadata = {
        "chapter_number": chapter_number,
        "beats": ["Beat 1", "Beat 2"],
        "scenes": [
            {"position": 0, "beat": "Beat 1"},
            {"position": 1, "beat": "Beat 2"}
        ]
    }
    
    # Save chapter
    stitcher.save_chapter(chapter_text, chapter_number, metadata)
    
    # Verify files were created
    assert (temp_story_dir / "chapters" / f"chapter_{chapter_number}.txt").exists()
    assert (temp_story_dir / "chapters" / f"chapter_{chapter_number}_metadata.json").exists()
    
    # Load chapter
    loaded = stitcher.load_chapter(chapter_number)
    assert loaded["text"] == chapter_text
    assert loaded["metadata"] == metadata

def test_list_chapters(temp_story_dir):
    """Test listing available chapters."""
    stitcher = ChapterStitcher(temp_story_dir)
    
    # Create some test chapters
    chapters_dir = temp_story_dir / "chapters"
    chapters_dir.mkdir()
    
    # Create valid chapter files
    (chapters_dir / "chapter_1.txt").touch()
    (chapters_dir / "chapter_2.txt").touch()
    
    # Create some invalid files that should be ignored
    (chapters_dir / "not_a_chapter.txt").touch()
    (chapters_dir / "chapter_invalid.txt").touch()
    
    chapters = stitcher.list_chapters()
    assert len(chapters) == 2
    assert 1 in chapters
    assert 2 in chapters
    assert sorted(chapters) == chapters  # Should be sorted 