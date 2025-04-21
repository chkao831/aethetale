from typing import List, Dict, Iterator
import json
from pathlib import Path

class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _find_sentence_boundary(self, text: str, position: int, direction: str = 'forward') -> int:
        """
        Find the nearest sentence boundary from a position.
        
        Args:
            text: Text to search in
            position: Starting position
            direction: 'forward' or 'backward'
            
        Returns:
            Position of the nearest sentence boundary
        """
        sentence_endings = ['.', '!', '?']
        
        if direction == 'forward':
            # Search forward for the next sentence ending
            for i in range(position, min(len(text), position + 100)):
                if text[i] in sentence_endings:
                    return i + 1
            return position
        else:
            # Search backward for the previous sentence ending
            for i in range(position - 1, max(-1, position - 100), -1):
                if i >= 0 and text[i] in sentence_endings:
                    return i + 1
            return position

    def split_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into chunks with metadata, processing one chunk at a time.
        
        Args:
            text: Input text to split
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate the initial end position
            end = min(start + self.chunk_size, len(text))
            
            # Find a proper sentence boundary for the end
            if end < len(text):
                end = self._find_sentence_boundary(text, end, 'backward')
            
            # Create the chunk
            chunk = {
                'text': text[start:end],
                'start_pos': start,
                'end_pos': end
            }
            chunks.append(chunk)
            
            # Calculate the next start position
            start = max(end - self.chunk_overlap, end)
            
            # If we're overlapping, find a proper sentence boundary
            if start < end:
                start = self._find_sentence_boundary(text, start, 'forward')
            
            # Break if we can't make progress
            if start >= len(text) or start <= chunks[-1]['start_pos']:
                break
        
        return chunks

    def save_chunks(self, chunks: List[Dict[str, str]], output_path: Path):
        """
        Save chunks to a JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save the chunks
        """
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)

    def load_chunks(self, input_path: Path) -> List[Dict[str, str]]:
        """
        Load chunks from a JSON file.
        
        Args:
            input_path: Path to load chunks from
            
        Returns:
            List of chunk dictionaries
        """
        with open(input_path, 'r') as f:
            return json.load(f) 