import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

import json
from pathlib import Path
from openai import OpenAI
from src.py_libs.ingestion.story_setup import StorySetup
from src.py_libs.flow.config_loader import ConfigLoader
from src.py_libs.flow.prompt_builder import PromptBuilder
from src.py_libs.flow.retriever import ContextRetriever
from src.py_libs.flow.generator import StoryGenerator
from src.py_libs.flow.chapter_stitcher import ChapterStitcher
from src.py_libs.flow.character_manager import CharacterManager

def main():
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
    story_path = Path("stories/the_clockwork_garden")
    if not story_path.exists():
        raise FileNotFoundError(f"Story path {story_path} does not exist")

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
        story_file = story_path / "content" / "story.txt"
        with open(story_file, "r") as f:
            story_text = f.read()
        print(f"✅ Read story file ({len(story_text)} characters)")

        # Process the story
        print("\nProcessing story...")
        
        # Debug: Print the prompt being used
        with open(story_path / "prompt.yaml", 'r') as f:
            import yaml
            prompts = yaml.safe_load(f)
            print("\nUsing prompt template:")
            print(prompts['story_analysis'])
        
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
            print(response.choices[0].message.content)
            
            # Try to parse the response
            try:
                elements = json.loads(response.choices[0].message.content)
                print("\nSuccessfully parsed JSON:")
                print(json.dumps(elements, indent=2))
                # Save the elements directly
                with open(story_path / "story_elements.json", 'w') as f:
                    json.dump(elements, f, indent=2)
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
        character_manager = CharacterManager(story_path, openai_client)
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
        config_loader = ConfigLoader(story_path, openai_client=openai_client)
        prompt_builder = PromptBuilder(story_path)
        retriever = ContextRetriever(story_path)
        generator = StoryGenerator(story_path, openai_client)
        stitcher = ChapterStitcher(story_path)
        print("✅ Flow components initialized")

        # Load configuration and beats
        config = config_loader.load_config()
        beat_descriptions = config_loader.load_beats()
        print("✅ Configuration and beats loaded")

        # Process each beat
        scenes = []
        for i, beat_description in enumerate(beat_descriptions, 1):
            print(f"\nProcessing beat: {beat_description}")
            
            # Analyze the beat using LLM
            beat = config_loader.analyze_beat(beat_description)
            beat['position'] = i
            print(f"✅ Beat analyzed: {beat}")
            
            # Get character profile and context
            character_name = beat['character']
            character_profile = retriever.get_character_profile(character_name)
            if character_profile:
                print(f"\nCharacter Profile for {character_name}:")
                print(f"Role: {character_profile.role}")
                print(f"Traits: {', '.join(character_profile.personality_traits)}")
                print(f"Goals: {', '.join(character_profile.goals)}")
                
            character_context = retriever.get_character_context(character_name)
            print(f"✅ Retrieved context for {character_name}")

            # Build prompt for the beat
            prompt = prompt_builder.build_beat_prompt(
                beat=beat['name'],
                context=character_context,
                style=beat['style']
            )
            print("✅ Prompt built")

            # Generate the text
            generated_text = generator.generate_text(prompt)
            print("✅ Text generated")

            # Add to scenes
            scenes.append({
                "text": generated_text,
                "position": beat['position'],
                "beat": beat['name']
            })

        # Stitch all scenes into a chapter
        chapter_text = stitcher.stitch_scenes(scenes)
        print("✅ Scenes stitched into chapter")

        # Save the chapter
        stitcher.save_chapter(
            chapter_text,
            chapter_number=1,
            metadata={
                "chapter_number": 1,
                "beats": beat_descriptions,
                "scenes": scenes
            }
        )
        print("✅ Chapter saved")

        # Print the generated chapter
        print("\nGenerated Chapter:")
        print("=" * 80)
        print(chapter_text)
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error during flow: {e}", file=sys.stderr)
        raise

    print("\n✨ Flow completed successfully!")

if __name__ == "__main__":
    main()
