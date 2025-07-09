import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import logging
import uuid

logger = logging.getLogger(__name__)

#Handler for Qdrant vector database operations
class QdrantDatabase:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "nawatech_faq"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self._connect()
    
    #connect qdrant
    def _connect(self):
        try:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info("Connected to Qdrant successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            # For development, try in-memory mode
            logger.info("Trying in-memory mode...")
            self.client = QdrantClient(":memory:")
    
    #create collection
    def create_collection(self, dimension: int):
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    #Add documents with embeddings to the collection
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        try:
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Ensure embedding is 1D
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=doc
                )
                points.append(point)
            
            logger.info(f"Adding {len(points)} documents to collection")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info("Documents added successfully")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    #search
    def search(self, query_embedding: np.ndarray, limit: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        try:
            # Ensure embedding is 1D array and convert to list
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            query_vector = query_embedding.tolist()
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for point in search_result:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []
    
    #get information
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": info.status,
                "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                "indexed_vectors_count": info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else 0,
                "points_count": info.points_count if hasattr(info, 'points_count') else 0
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "name": self.collection_name,
                "status": "unknown",
                "vectors_count": 0,
                "indexed_vectors_count": 0,
                "points_count": 0
            }
    
    #delete collection
    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    #load knowledge faq data csv
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