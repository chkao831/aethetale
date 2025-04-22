import os
import sys
import argparse
import yaml
import json
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from pathlib import Path
from openai import OpenAI
from src.py_libs.ingestion.story_setup import StorySetup
from src.py_libs.flow.config_loader import ConfigLoader
from src.py_libs.flow.prompt_builder import PromptBuilder
from src.py_libs.flow.retriever import ContextRetriever
from src.py_libs.flow.generator import StoryGenerator
from src.py_libs.flow.chapter_stitcher import ChapterStitcher
from src.py_libs.flow.character_manager import CharacterManager
from src.py_libs.ui.story_creator import StoryCreator

def get_story_file(story_path):
    """Find the story file in either story.md or content/story.txt"""
    # Try story.md first
    story_md = story_path / "story.md"
    if story_md.exists():
        return story_md
    
    # Try content/story.txt
    story_txt = story_path / "content" / "story.txt"
    if story_txt.exists():
        return story_txt
    
    # If neither exists, raise error
    raise FileNotFoundError(
        f"Could not find story file in {story_path}. "
        f"Expected either story.md or content/story.txt"
    )

def main():
    parser = argparse.ArgumentParser(description="Story Generation System")
    parser.add_argument("--create", action="store_true", help="Create a new story")
    parser.add_argument("--story", type=str, default="the_clockwork_garden", 
                       help="Name of the story to process (default: the_clockwork_garden)")
    parser.add_argument("--language", type=str, choices=["auto", "en", "zh"], default="auto",
                       help="Language for generated content (auto/en/zh)")
    parser.add_argument("--model", type=str, choices=["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo",
                       help="Model to use for generation (default: gpt-3.5-turbo)")
    
    args = parser.parse_args()

    if args.create:
        StoryCreator().create_new_story()
        return

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running.")

    # Initialize OpenAI client
    try:
        openai_client = OpenAI()
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}", file=sys.stderr)
        raise

    # Set up paths
    story_path = Path("stories") / args.story
    shared_config_path = Path("config/shared")
    
    if not story_path.exists():
        raise FileNotFoundError(f"Story path {story_path} does not exist. Available stories: {', '.join([d.name for d in Path('stories').iterdir() if d.is_dir()])}")

    print(f"\n📚 Processing story at: {story_path}")

    # ===== INGESTION PHASE =====
    print("\n=== INGESTION PHASE ===")
    
    try:
        # Initialize story setup
        story_setup = StorySetup(story_path, openai_client=openai_client)
        # Use a smaller model for testing
        story_setup.embedder.model_name = "paraphrase-MiniLM-L3-v2"
        print("✅ StorySetup initialized")

        # Read the story text
        story_file = get_story_file(story_path)
        print(f"✅ Found story file at: {story_file}")
        with open(story_file, "r", encoding='utf-8') as f:
            story_text = f.read()
        print(f"✅ Read story file ({len(story_text)} characters)")

        # Determine language if auto
        if args.language == "auto":
            # Simple heuristic: if more than 50% of characters are Chinese, use Chinese
            chinese_chars = sum(1 for c in story_text if '\u4e00' <= c <= '\u9fff')
            args.language = "zh" if chinese_chars / len(story_text) > 0.5 else "en"
            print(f"Auto-detected language: {args.language}")

        print(f"Using specified model: {args.model}")

        # Load shared configuration
        with open(shared_config_path / "prompt.yaml", 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        with open(shared_config_path / "config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ Loaded shared configuration")

        # Process the story
        print("\nProcessing story...")
        
        # Try story analysis with error handling
        try:
            story_setup.analyzer.update_story_elements(story_text)
            print("✅ Story elements analyzed")
        except Exception as e:
            print(f"❌ Error during story analysis: {e}")
            print("\nAttempting to analyze with direct API call...")
            
            # Try direct API call for debugging
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a story analysis assistant."},
                    {"role": "user", "content": prompts['story_analysis'].format(text=story_text)}
                ],
                temperature=0.3
            )
            
            print("\nRaw LLM response:")
            content = response.choices[0].message.content.strip()
            print(content)
            
            # Try to parse the response
            try:
                # Remove the ```json and ``` markers if they exist
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                elements = json.loads(content)
                print("\nSuccessfully parsed JSON:")
                print(json.dumps(elements, indent=2, ensure_ascii=False))
                # Save the elements directly
                with open(story_path / "story_elements.json", 'w', encoding='utf-8') as f:
                    json.dump(elements, f, indent=2, ensure_ascii=False)
                print("✅ Story elements saved")
            except json.JSONDecodeError as je:
                print(f"❌ Failed to parse JSON: {je}")
                raise

        chunks = story_setup.splitter.split_text(story_text)
        print(f"✅ Text split into {len(chunks)} chunks")

        chunks_with_embeddings = story_setup.embedder.embed_chunks(chunks)
        print("✅ Embeddings generated")

        story_setup.index_builder.build_index(chunks_with_embeddings)
        print("✅ Index built")

        version_id = story_setup.version_manager.create_version("Initial version")
        story_setup._save_artifacts(chunks_with_embeddings, version_id)
        print(f"✅ Version created and artifacts saved (Version ID: {version_id})")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}", file=sys.stderr)
        raise

    # ===== CHARACTER PROFILE EXTRACTION =====
    print("\n=== CHARACTER PROFILE EXTRACTION ===")
    
    try:
        # Initialize character manager
        character_manager = CharacterManager(story_path, openai_client, model=args.model)
        print("✅ CharacterManager initialized")
        
        # Extract and update character profiles
        character_manager.update_character_profiles(story_text)
        print("✅ Character profiles extracted and updated")
        
        # Print character network
        network = character_manager.get_character_network()
        print("\nCharacter Relationship Network:")
        for character, related in network.items():
            print(f"\n{character}:")
            for relation in related:
                print(f"  - {relation}")
                
    except Exception as e:
        print(f"❌ Error during character profile extraction: {e}", file=sys.stderr)
        raise

    # ===== FLOW PHASE =====
    print("\n=== FLOW PHASE ===")
    
    try:
        # Initialize flow components
        print("\nInitializing flow components...")
        config_loader = ConfigLoader(story_path, openai_client=openai_client, model=args.model)
        prompt_builder = PromptBuilder(story_path, language=args.language)
        retriever = ContextRetriever(story_path)
        generator = StoryGenerator(story_path, openai_client, language=args.language, model=args.model)
        stitcher = ChapterStitcher(story_path)
        print("✅ Flow components initialized")

        # Load configuration and beats
        config = config_loader.load_config()
        beat_descriptions = config_loader.load_beats()
        print("✅ Configuration and beats loaded")

        # Get story style and context
        story_elements = story_setup.analyzer.load_story_elements()
        story_style = story_elements['style']
        # Use larger context window for Chinese content
        context_window = 3000 if args.language == "zh" else 1000
        story_context = story_text[-context_window:]  # Use last N chars as context
        
        # Get character context for each character in the story
        character_contexts = {}
        for character_name in story_elements['characters'].keys():
            character_contexts[character_name] = retriever.get_character_context(character_name)
        
        # Build prompt for continuous narrative with character context
        prompt = prompt_builder.build_beat_prompt(
            beats="\n".join(f"- {beat}" for beat in beat_descriptions),
            context=story_context,
            style=story_style,
            character_contexts=character_contexts
        )
        print("✅ Prompt built")

        # Generate the continuous narrative
        generated_text = generator.generate_text(prompt)
        print("✅ Text generated")

        # Save the chapter
        stitcher.save_chapter(
            generated_text,
            chapter_number=1,
            metadata={
                "chapter_number": 1,
                "beats": beat_descriptions,
                "language": args.language
            }
        )
        print("✅ Chapter saved")

        # Print the generated chapter
        print("\nGenerated Chapter:")
        print("=" * 80)
        print(generated_text)
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error during flow: {e}", file=sys.stderr)
        raise

    print("\n✨ Flow completed successfully!")

if __name__ == "__main__":
    main()
