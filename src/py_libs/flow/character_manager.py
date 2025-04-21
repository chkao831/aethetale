from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime
from openai import OpenAI
from src.py_libs.models.character_profile import CharacterProfile, FamilyRelation
from pydantic import ValidationError

class CharacterManager:
    def __init__(self, story_path: Path, openai_client: OpenAI | None = None):
        """
        Initialize the character manager.
        
        Args:
            story_path: Path to the story directory
            openai_client: OpenAI client instance to use, if None a new one will be created
        """
        self.story_path = story_path
        self.client = openai_client if openai_client is not None else OpenAI()
        self.profiles_path = story_path / "character_profiles.json"
        
    def _serialize_datetime(self, dt: datetime) -> str:
        """Convert datetime to ISO format string."""
        return dt.isoformat()
        
    def _serialize_profile(self, profile: CharacterProfile) -> dict:
        """Serialize a CharacterProfile to a JSON-compatible dictionary."""
        data = profile.model_dump()
        # Convert datetime objects to strings
        data['created_at'] = self._serialize_datetime(profile.created_at)
        data['updated_at'] = self._serialize_datetime(profile.updated_at)
        return data
        
    def extract_character_profiles(self, text: str) -> Dict[str, CharacterProfile]:
        """
        Extract character profiles from story text using LLM.
        
        Args:
            text: The story text to analyze
            
        Returns:
            Dictionary mapping character names to their profiles
        """
        prompt = f"""
        Analyze the following story text and extract detailed character profiles.
        For each character mentioned, create a rich profile with the following information:
        
        - name: Canonical name
        - aliases: List of other names or titles
        - role: Narrative or mythic role
        - occupation: What they do
        - personality_traits: Core emotional/behavioral traits
        - goals: Personal motivations
        - fears: Deepest anxieties
        - lovers: Romantic links
        - friends: Ally links
        - enemies: Conflicts
        - family: List of family relationships, each with relation_type and name
        - key_events: Important scenes they appear in
        
        Format the output as a JSON object where each key is a character name
        and the value is their profile information. Make sure the output is valid JSON.
        
        Example format:
        {{
            "CharacterName": {{
                "name": "CharacterName",
                "aliases": ["Alias1", "Alias2"],
                "role": "Protagonist",
                "occupation": "Occupation",
                "personality_traits": ["Trait1", "Trait2"],
                "goals": ["Goal1", "Goal2"],
                "fears": ["Fear1", "Fear2"],
                "lovers": ["Lover1"],
                "friends": ["Friend1", "Friend2"],
                "enemies": ["Enemy1"],
                "family": [
                    {{
                        "relation_type": "parent",
                        "name": "ParentName"
                    }}
                ],
                "key_events": ["Event1", "Event2"]
            }}
        }}
        
        Story text:
        {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a literary analyst specializing in character development. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Get the response content
            content = response.choices[0].message.content.strip()
            
            # Try to parse the JSON
            try:
                raw_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Raw LLM response: {content}")
                raise Exception(f"Failed to parse JSON response: {str(e)}")
                
            # Convert to CharacterProfile objects with validation
            profiles = {}
            for name, data in raw_data.items():
                try:
                    # Ensure family relations are properly structured
                    if "family" in data:
                        data["family"] = [
                            rel if isinstance(rel, dict) else {"relation_type": "unknown", "name": rel}
                            for rel in data["family"]
                        ]
                    
                    # Create and validate the profile
                    profile = CharacterProfile(**data)
                    profiles[name] = profile
                except ValidationError as e:
                    print(f"Validation error for character {name}: {str(e)}")
                    continue
                    
            return profiles
            
        except Exception as e:
            raise Exception(f"Error extracting character profiles: {str(e)}")
            
    def update_character_profiles(self, text: str):
        """
        Update character profiles based on new story text.
        
        Args:
            text: New story text to analyze
        """
        # Extract new profiles
        new_profiles = self.extract_character_profiles(text)
        
        # Load existing profiles if they exist
        existing_profiles = {}
        if self.profiles_path.exists():
            try:
                with open(self.profiles_path, 'r') as f:
                    existing_data = json.load(f)
                    for name, data in existing_data.items():
                        try:
                            existing_profiles[name] = CharacterProfile(**data)
                        except ValidationError as e:
                            print(f"Validation error for existing character {name}: {str(e)}")
                            continue
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Error loading existing profiles: {str(e)}")
                
        # Merge profiles
        for name, new_profile in new_profiles.items():
            if name in existing_profiles:
                # Update existing profile
                existing_profile = existing_profiles[name]
                # Merge lists with validation
                try:
                    merged_data = {
                        "name": name,
                        "aliases": list(set(existing_profile.aliases + new_profile.aliases)),
                        "role": new_profile.role if len(new_profile.role) > len(existing_profile.role) else existing_profile.role,
                        "occupation": new_profile.occupation if len(new_profile.occupation) > len(existing_profile.occupation) else existing_profile.occupation,
                        "personality_traits": list(set(existing_profile.personality_traits + new_profile.personality_traits)),
                        "goals": list(set(existing_profile.goals + new_profile.goals)),
                        "fears": list(set(existing_profile.fears + new_profile.fears)),
                        "lovers": list(set(existing_profile.lovers + new_profile.lovers)),
                        "friends": list(set(existing_profile.friends + new_profile.friends)),
                        "enemies": list(set(existing_profile.enemies + new_profile.enemies)),
                        "family": list({
                            (rel.relation_type, rel.name): rel
                            for rel in (existing_profile.family + new_profile.family)
                        }.values()),
                        "key_events": list(set(existing_profile.key_events + new_profile.key_events)),
                        "profile_text": new_profile.profile_text if (new_profile.profile_text and len(new_profile.profile_text) > len(existing_profile.profile_text or "")) else existing_profile.profile_text,
                        "style_embedding": new_profile.style_embedding or existing_profile.style_embedding,
                        "created_at": existing_profile.created_at,
                        "updated_at": datetime.now()
                    }
                    existing_profiles[name] = CharacterProfile(**merged_data)
                except ValidationError as e:
                    print(f"Error merging profiles for {name}: {str(e)}")
                    continue
            else:
                # Add new profile
                existing_profiles[name] = new_profile
                
        # Save updated profiles
        try:
            profiles_data = {
                name: self._serialize_profile(profile)
                for name, profile in existing_profiles.items()
            }
            with open(self.profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
        except Exception as e:
            raise Exception(f"Error saving profiles: {str(e)}")
            
    def get_character_network(self) -> Dict[str, List[str]]:
        """
        Generate a character relationship network.
        
        Returns:
            Dictionary mapping character names to lists of related characters
        """
        if not self.profiles_path.exists():
            return {}
            
        try:
            with open(self.profiles_path, 'r') as f:
                profiles_data = json.load(f)
                
            network = {}
            for name, data in profiles_data.items():
                try:
                    profile = CharacterProfile(**data)
                    related = (
                        profile.lovers +
                        profile.friends +
                        profile.enemies +
                        [rel.name for rel in profile.family]
                    )
                    network[name] = list(set(related))
                except ValidationError as e:
                    print(f"Error processing network for {name}: {str(e)}")
                    continue
                    
            return network
        except Exception as e:
            print(f"Error generating character network: {str(e)}")
            return {} 