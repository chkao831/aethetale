from typing import List, Dict, Any, Iterator
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import os

class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 8):
        """
        Initialize the text embedder with a SentenceTransformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            batch_size: Number of texts to process at once
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        
    def _batch_iterator(self, texts: List[str]) -> Iterator[List[str]]:
        """
        Create batches of texts.
        
        Args:
            texts: List of text strings to batch
            
        Returns:
            Iterator of text batches
        """
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts in batches.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        for batch in self._batch_iterator(texts):
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to chunk dictionaries in batches.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            List of chunk dictionaries with added 'embedding' field
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
            
        return chunks
    
    def save_embeddings(self, chunks: List[Dict[str, Any]], output_path: Path):
        """
        Save chunks with embeddings to a JSON file.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
            output_path: Path to save the embeddings
        """
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)
            
    def load_embeddings(self, input_path: Path) -> List[Dict[str, Any]]:
        """
        Load chunks with embeddings from a JSON file.
        
        Args:
            input_path: Path to load embeddings from
            
        Returns:
            List of chunk dictionaries with embeddings
        """
        with open(input_path, 'r') as f:
            return json.load(f) 