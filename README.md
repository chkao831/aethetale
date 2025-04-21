## Quick start
`python -m src.main`


## Pre-commit
`pre-commit install`

## Setting up a new story

To set up a new story, follow these steps:

1. Create a new directory under `stories/` with your story name:
   ```
   stories/
   └── my_new_story/
   ```

2. Create the required files:
   - `story.md`: The main story content
   - `beats.yaml`: List of story beats in chronological order

3. Example `beats.yaml` format:
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
