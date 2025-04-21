## Quick start
```bash
# Run with default story (the_clockwork_garden)
python -m src.main

# Run with specific story
python -m src.main --story Lucent

# Run with specific language
python -m src.main --story Lucent --language zh
```

## Running Tests
To run the test suite:
```bash
python -m pytest src/tests -v
```

## Pre-commit
`pre-commit install`

## Command Line Options

### 1. Creating a New Story
To create a new story using the interactive UI:
```bash
python -m src.main --create
```

The UI will guide you through the following steps:
1. Enter your story name
2. Create the story directory structure
3. Write your story content in `story.md`
4. Define your story beats in `beats.yaml`

### 2. Processing Your Story
Once you've created your story, you can process it with these options:

1. `--story`: Specify which story to process
   ```bash
   python -m src.main --story your_story_name
   ```

2. `--language`: Choose the language for generated content
   - "auto": Auto-detect based on story content (default)
   - "en": English
   - "zh": Chinese
   ```bash
   python -m src.main --story your_story_name --language zh
   ```

The system uses shared configurations from `configs/shared/` for consistent behavior across all stories.

## Story Structure

Your story directory should contain:
```
stories/
└── your_story_name/
    ├── story.md    # Your story content
    └── beats.yaml  # List of story beats
```

Example `beats.yaml` format:
```yaml
beats:
  - "Introduce the protagonist and their initial state"
  - "Protagonist discovers a mysterious artifact"
  - "Introduce the mentor character"
```

The system will automatically:
- Analyze each beat to determine appropriate characters, context, and style
- Generate coherent scenes based on the story content
- Stitch the scenes together into a flowing narrative

Note: Keep beat descriptions simple and focused on key story moments. The AI will handle the details of character interactions, tone, and narrative style.
