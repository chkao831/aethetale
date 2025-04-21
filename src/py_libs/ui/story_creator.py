from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.markdown import Markdown
from pathlib import Path
import yaml
import os

console = Console()

class StoryCreator:
    def __init__(self):
        self.stories_dir = Path("stories")
        self.stories_dir.mkdir(exist_ok=True)

    def create_new_story(self):
        console.print(Panel.fit("📚 Welcome to Story Creator! Let's create a new story.", style="bold blue"))
        
        # Get story name
        story_name = Prompt.ask("What would you like to name your story?")
        story_dir = self.stories_dir / story_name
        story_dir.mkdir(exist_ok=True)

        # Create story.md
        console.print("\n[bold]Let's write your story:[/bold]")
        console.print("(Press Enter twice to finish writing)")
        story_content = []
        while True:
            line = input()
            if not line and story_content and not story_content[-1]:
                break
            story_content.append(line)
        
        story_file = story_dir / "story.md"
        with open(story_file, "w", encoding='utf-8') as f:
            f.write("\n".join(story_content))

        # Create beats.yaml
        console.print("\n[bold]Now, let's outline the key beats of your story:[/bold]")
        console.print("(Enter each beat on a new line. Press Enter twice to finish)")
        
        beats = []
        while True:
            beat = input()
            if not beat and beats and not beats[-1]:
                break
            beats.append(beat)

        beats_data = {"beats": [b for b in beats if b]}
        beats_file = story_dir / "beats.yaml"
        with open(beats_file, "w", encoding='utf-8') as f:
            yaml.dump(beats_data, f, 
                     default_flow_style=False,
                     allow_unicode=True,  # This ensures Chinese characters are written directly
                     encoding='utf-8')    # This ensures proper UTF-8 encoding

        console.print(Panel.fit(
            f"✅ Story created successfully!\n"
            f"Location: {story_dir}\n"
            f"Files created:\n"
            f"- {story_file}\n"
            f"- {beats_file}",
            style="bold green"
        ))

        if Confirm.ask("\nWould you like to review your story?"):
            self.review_story(story_dir)

    def review_story(self, story_dir):
        console.print("\n[bold]Your Story Content:[/bold]")
        with open(story_dir / "story.md", "r", encoding='utf-8') as f:
            story_content = f.read()
            console.print(Markdown(story_content))

        console.print("\n[bold]Story Beats:[/bold]")
        with open(story_dir / "beats.yaml", "r", encoding='utf-8') as f:
            beats_data = yaml.safe_load(f)
            for i, beat in enumerate(beats_data["beats"], 1):
                console.print(f"{i}. {beat}")

def main():
    creator = StoryCreator()
    creator.create_new_story()

if __name__ == "__main__":
    main() 