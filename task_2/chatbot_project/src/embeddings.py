import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

#Handler for text embeddings using SentenceTransformers
class EmbeddingHandler:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    #load model
    def _load_model(self):
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    #encode text to embed
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    #get dimension
    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    #count cosine similiarity
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0