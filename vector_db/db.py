"""GPU-accelerated vector database."""
import numpy as np
from typing import List, Dict, Any, Optional

class VectorDB:
    def __init__(self, dim=1536, index_type="hnsw", device="cuda"):
        self.dim = dim
        self.index_type = index_type
        self.device = device
        self.vectors = []
        self.metadata = []
        
    def insert(self, vectors, metadata=None):
        self.vectors.extend(vectors)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(vectors))
            
    def search(self, query, top_k=10, filter_metadata=None):
        if not self.vectors:
            return []
        scores = [float(np.dot(query, v) / (np.linalg.norm(query) * np.linalg.norm(v) + 1e-8)) 
                  for v in self.vectors]
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"id": i, "score": scores[i], "metadata": self.metadata[i]} for i in indices]
        
    def delete(self, ids):
        for i in sorted(ids, reverse=True):
            if i < len(self.vectors):
                del self.vectors[i]
                del self.metadata[i]
                
    def count(self):
        return len(self.vectors)
