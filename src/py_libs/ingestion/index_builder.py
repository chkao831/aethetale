from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import json
import os
import faiss

class IndexBuilder:
    def __init__(self, dimension: int = 384):  # Default dimension for all-MiniLM-L6-v2
        """
        Initialize the index builder.
        
        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        
    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build a FAISS index from chunk embeddings.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
        self.index.add(embeddings)
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[int]:
        """
        Search the index for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of indices of similar chunks
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return indices[0].tolist()
    
    def save_index(self, output_path: Path):
        """
        Save the FAISS index to disk.
        
        Args:
            output_path: Path to save the index
        """
        os.makedirs(output_path.parent, exist_ok=True)
        faiss.write_index(self.index, str(output_path))
        
    def load_index(self, input_path: Path):
        """
        Load a FAISS index from disk.
        
        Args:
            input_path: Path to load the index from
        """
        self.index = faiss.read_index(str(input_path))
        
    def save_metadata(self, chunks: List[Dict[str, Any]], output_path: Path):
        """
        Save chunk metadata to a JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save the metadata
        """
        metadata = [{
            'text': chunk['text'],
            'start_pos': chunk['start_pos'],
            'end_pos': chunk['end_pos']
        } for chunk in chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
    def load_metadata(self, input_path: Path) -> List[Dict[str, Any]]:
        """
        Load chunk metadata from a JSON file.
        
        Args:
            input_path: Path to load metadata from
            
        Returns:
            List of chunk dictionaries
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f) 