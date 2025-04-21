from typing import List, Dict, Any
from pathlib import Path
import json
import yaml
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
        
        # Load prompts from shared config
        shared_config_path = Path("config/shared")
        with open(shared_config_path / "prompt.yaml", 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
        
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
        
    def _convert_family_relation(self, relation: str | dict) -> Dict[str, str]:
        """
        Convert a family relation to the proper format.
        
        Args:
            relation: String or dict representing a family relation
            
        Returns:
            Dictionary with name and relation_type fields
        """
        if isinstance(relation, dict):
            # If it's already a dict, ensure it has the required fields
            if 'name' not in relation:
                return {
                    'name': 'Unknown',
                    'relation_type': 'Unknown'
                }
            # Convert relation to relation_type if needed
            return {
                'name': relation['name'],
                'relation_type': relation.get('relation_type', relation.get('relation', 'Unknown'))
            }
        else:
            # If it's a string, use it as the name and set relation_type as Unknown
            return {
                'name': str(relation),
                'relation_type': 'Unknown'
            }
            
    def _convert_family_list(self, family_data: list | dict) -> list:
        """
        Convert family data to a list of properly formatted family relations.
        
        Args:
            family_data: List or dict of family relations
            
        Returns:
            List of properly formatted family relations
        """
        if isinstance(family_data, dict):
            # Convert dict to list of relations
            return [
                {
                    'name': name,
                    'relation_type': data.get('relation_type', data.get('relation', 'Unknown'))
                }
                for name, data in family_data.items()
            ]
        elif isinstance(family_data, list):
            # Convert each item in the list
            return [self._convert_family_relation(item) for item in family_data]
        else:
            return []
            
    def extract_character_profiles(self, text: str) -> Dict[str, CharacterProfile]:
        """
        Extract character profiles from the text using LLM.
        
        Args:
            text: The story text to analyze
            
        Returns:
            Dictionary mapping character names to their profiles
        """
        # Get the character analysis prompt from loaded prompts
        character_prompt = self.prompts.get('character_extraction', """
        Analyze the following text and extract character profiles. For each character, provide:
        - Name
        - Role in the story
        - Occupation
        - Personality traits
        - Goals and motivations
        - Fears and weaknesses
        - Relationships (family, friends, enemies, lovers)
        - Key events they're involved in
        
        Text: {text}
        
        Provide the analysis in JSON format with the following structure for each character:
        {{
            "character_name": {{
                "name": "string",
                "aliases": ["string"],
                "role": "string",
                "occupation": "string",
                "personality_traits": ["string"],
                "goals": ["string"],
                "fears": ["string"],
                "lovers": ["string"],
                "friends": ["string"],
                "enemies": ["string"],
                "family": [{{ "name": "string", "relation_type": "string" }}],
                "key_events": ["string"]
            }}
        }}
        
        Note: For family relationships, each entry must include both name and relation_type. Example:
        "family": [
            {{"name": "John", "relation_type": "Father"}},
            {{"name": "Mary", "relation_type": "Sister"}}
        ]
        """)
        
        # Format the prompt
        formatted_prompt = character_prompt.format(text=text)
        
        # Call the LLM
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a character analysis assistant. Always respond with valid JSON. For family relationships, always include both name and relation_type fields."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Get the response content
        content = response.choices[0].message.content
        print(f"Raw LLM response:\n{content}")
        
        # Try to parse the JSON response
        try:
            # First try direct parsing
            profiles_data = json.loads(content)
        except json.JSONDecodeError as e:
            # If direct parsing fails, try to extract JSON from the response
            try:
                # Look for JSON content between ```json and ``` markers
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    profiles_data = json.loads(json_match.group(1))
                else:
                    # If no markers found, try to find the first valid JSON object
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        profiles_data = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No valid JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse character profiles from LLM response: {str(e)}")
                
        # Process the profiles
        profiles = {}
        for name, data in profiles_data.items():
            # Convert family data to proper format
            if 'family' in data:
                data['family'] = self._convert_family_list(data['family'])
                
            # Convert other list fields to ensure they're strings
            for field in ['aliases', 'personality_traits', 'goals', 'fears', 'lovers', 'friends', 'enemies', 'key_events']:
                if field in data and isinstance(data[field], list):
                    data[field] = [str(item) for item in data[field]]
                    
            # Create the profile
            try:
                profiles[name] = CharacterProfile(
                    name=name,
                    aliases=data.get('aliases', []),
                    role=data.get('role', ''),
                    occupation=data.get('occupation', ''),
                    personality_traits=data.get('personality_traits', []),
                    goals=data.get('goals', []),
                    fears=data.get('fears', []),
                    lovers=data.get('lovers', []),
                    friends=data.get('friends', []),
                    enemies=data.get('enemies', []),
                    family=data.get('family', []),
                    key_events=data.get('key_events', [])
                )
            except Exception as e:
                print(f"Error creating profile for {name}: {str(e)}")
                print(f"Profile data: {data}")
                raise
                
        return profiles
            
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
                with open(self.profiles_path, 'r', encoding='utf-8') as f:
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
            with open(self.profiles_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
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