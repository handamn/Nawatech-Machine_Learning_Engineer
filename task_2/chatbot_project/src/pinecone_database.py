import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import uuid
import time

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not installed. Use: pip install pinecone-client")

logger = logging.getLogger(__name__)

#Handler for Pinecone
class PineconeDatabase:
    def __init__(self, api_key: str, index_name: str = "nawatech-faq", environment: str = "gcp-starter"):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available. Install with: pip install pinecone-client")
        
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.pc = None
        self.index = None
        self._connect()
    
    #connect pinecone
    def _connect(self):
        try:
            logger.info(f"Connecting to Pinecone environment: {self.environment}")
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists, create if not
            self._ensure_index_exists()
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info("Connected to Pinecone successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise
    
    #check index exist
    def _ensure_index_exists(self):
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                logger.info("Index created successfully")
            else:
                logger.info(f"Index {self.index_name} already exists")
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise
    
    #add document
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        try:
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Ensure embedding is 1D
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                vector_id = str(uuid.uuid4())
                vector_data = {
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": {
                        "question": doc.get("question", ""),
                        "answer": doc.get("answer", ""),
                        "type": doc.get("type", "faq")
                    }
                }
                vectors.append(vector_data)
            
            logger.info(f"Adding {len(vectors)} documents to Pinecone index")
            # Upsert in batches of 100 (Pinecone limit)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info("Documents added successfully to Pinecone")
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            raise
    
    #search similiar document
    def search(self, query_embedding: np.ndarray, limit: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        try:
            # Ensure embedding is 1D array
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            query_vector = query_embedding.tolist()
            
            search_result = self.index.query(
                vector=query_vector,
                top_k=limit,
                include_metadata=True
            )
            
            results = []
            for match in search_result.matches:
                if match.score >= score_threshold:
                    result = {
                        "id": match.id,
                        "score": match.score,
                        "payload": {
                            "question": match.metadata.get("question", ""),
                            "answer": match.metadata.get("answer", ""),
                            "type": match.metadata.get("type", "faq")
                        }
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar documents in Pinecone")
            return results
        except Exception as e:
            logger.error(f"Failed to search in Pinecone: {e}")
            return []
    
    #get index info
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            return {
                "name": self.index_name,
                "status": "ready",
                "vectors_count": stats.total_vector_count,
                "indexed_vectors_count": stats.total_vector_count,
                "points_count": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone index info: {e}")
            return {
                "name": self.index_name,
                "status": "unknown",
                "vectors_count": 0,
                "indexed_vectors_count": 0,
                "points_count": 0
            }
    
    #delete
    def delete_collection(self):
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Pinecone index {self.index_name} deleted")
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index: {e}")
    
    #load faq data csv
    def load_faq_data(self, csv_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Loading FAQ data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Clean data
            df = df.dropna(subset=['Question', 'Answer'])
            df['Question'] = df['Question'].str.strip()
            df['Answer'] = df['Answer'].str.strip()
            
            # Convert to documents
            documents = []
            for _, row in df.iterrows():
                doc = {
                    "question": row['Question'],
                    "answer": row['Answer'],
                    "type": "faq"
                }
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} FAQ items")
            return documents
        except Exception as e:
            logger.error(f"Failed to load FAQ data: {e}")
            raise