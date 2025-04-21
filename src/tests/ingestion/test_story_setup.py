import os
import sys
import gc
from pathlib import Path
from openai import OpenAI
from src.py_libs.ingestion.story_setup import StorySetup

def test_story_setup():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running tests.")

    print("Debug: OpenAI API key is set")

    # Initialize OpenAI client
    try:
        openai_client = OpenAI()
        print("Debug: OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}", file=sys.stderr)
        raise

    # Initialize story setup
    story_path = Path("stories/the_clockwork_garden")
    if not story_path.exists():
        raise FileNotFoundError(f"Story path {story_path} does not exist")

    print(f"Debug: Story path exists at {story_path}")

    try:
        story_setup = StorySetup(story_path, openai_client=openai_client)
        # Override the default model with a smaller one
        story_setup.embedder.model_name = "paraphrase-MiniLM-L3-v2"
        print("Debug: StorySetup initialized successfully")
    except Exception as e:
        print(f"Error initializing StorySetup: {e}", file=sys.stderr)
        raise

    # Read the story text
    story_file = story_path / "content" / "story.txt"
    if not story_file.exists():
        raise FileNotFoundError(f"Story file {story_file} does not exist")

    print(f"Debug: Story file exists at {story_file}")

    try:
        with open(story_file, "r") as f:
            story_text = f.read()
        print(f"Debug: Successfully read story file ({len(story_text)} characters)")
    except Exception as e:
        print(f"Error reading story file: {e}", file=sys.stderr)
        raise

    # Process the story
    print("\nStarting story processing...")
    try:
        print("Debug: Starting story analysis...")
        story_setup.analyzer.update_story_elements(story_text)
        print("Debug: Story analysis complete")

        print("Debug: Starting text splitting...")
        chunks = story_setup.splitter.split_text(story_text)
        print(f"Debug: Text split into {len(chunks)} chunks")

        print("Debug: Starting embedding generation...")
        gc.collect()  # Force garbage collection before embedding
        chunks_with_embeddings = story_setup.embedder.embed_chunks(chunks)
        print("Debug: Embeddings generated successfully")

        print("Debug: Building index...")
        story_setup.index_builder.build_index(chunks_with_embeddings)
        print("Debug: Index built successfully")

        print("Debug: Creating version and saving artifacts...")
        version_id = story_setup.version_manager.create_version("Initial version")
        story_setup._save_artifacts(chunks_with_embeddings, version_id)
        print(f"Debug: Version created and artifacts saved. Version ID: {version_id}")

    except Exception as e:
        print(f"Error during story processing: {e}", file=sys.stderr)
        raise

    # Verify the output
    print("\nVerifying output files...")
    version_dir = story_path / "versions" / version_id
    faiss_index = version_dir / "faiss_index" / "index.faiss"
    metadata = version_dir / "faiss_index" / "metadata.json"
    passages = version_dir / "passages.json"
    story_elements = story_path / "story_elements.json"

    print(f"Version directory: {version_dir}")
    print(f"FAISS index exists: {faiss_index.exists()}")
    print(f"Metadata exists: {metadata.exists()}")
    print(f"Passages exists: {passages.exists()}")
    print(f"Story elements exists: {story_elements.exists()}")

    # Verify all required files exist
    required_files = [faiss_index, metadata, passages, story_elements]
    missing_files = [str(f) for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    try:
        test_story_setup()
    except Exception as e:
        print(f"\nTest failed with error: {e}", file=sys.stderr)
        sys.exit(1) 