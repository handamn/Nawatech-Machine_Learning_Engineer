import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

logger = logging.getLogger(__name__)

#Hybrid search engine combining semantic and keyword search
class HybridSearchEngine:
    def __init__(self, semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.tfidf_vectorizer = None
        self.document_texts = []
        self.documents = []
        
        # Validate weights
        if abs(semantic_weight + keyword_weight - 1.0) > 0.001:
            logger.warning(f"Weights don't sum to 1.0: {semantic_weight + keyword_weight}")
    
    #Index documents for keyword search
    def index_documents(self, documents: List[Dict[str, Any]]):
        try:
            logger.info(f"Indexing {len(documents)} documents for hybrid search")
            
            self.documents = documents
            
            # Prepare texts for TF-IDF
            self.document_texts = []
            for doc in documents:
                # Combine question and answer for better keyword matching
                text = f"{doc.get('question', '')} {doc.get('answer', '')}"
                processed_text = self._preprocess_text(text)
                self.document_texts.append(processed_text)
            
            # Create TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                lowercase=True,
                strip_accents='unicode'
            )
            
            # Fit vectorizer on documents
            self.tfidf_vectorizer.fit(self.document_texts)
            
            logger.info("Documents indexed successfully for hybrid search")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    #search using combine semantic and keyword
    def search(self, 
               query: str, 
               semantic_results: List[Dict[str, Any]], 
               limit: int = 5) -> List[Dict[str, Any]]:
        try:
            # Perform keyword search
            keyword_results = self._keyword_search(query, limit * 2)  # Get more for fusion
            
            # Combine and re-rank results
            hybrid_results = self._fuse_results(semantic_results, keyword_results, limit)
            
            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to semantic results only
            return semantic_results[:limit]
    
    #keyword search
    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        try:
            if not self.tfidf_vectorizer or not self.document_texts:
                return []
            
            # Preprocess query
            processed_query = self._preprocess_text(query)
            
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # Transform all documents
            doc_vectors = self.tfidf_vectorizer.transform(self.document_texts)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:limit]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include non-zero similarities
                    result = {
                        "id": f"keyword_{idx}",
                        "score": float(similarities[idx]),
                        "payload": self.documents[idx],
                        "search_type": "keyword"
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    #Fuse semantic and keyword results using weighted scoring
    def _fuse_results(self, 
                     semantic_results: List[Dict[str, Any]], 
                     keyword_results: List[Dict[str, Any]], 
                     limit: int) -> List[Dict[str, Any]]:

        try:
            # Create a dictionary to store combined scores
            combined_scores = {}
            result_data = {}
            
            # Process semantic results
            for result in semantic_results:
                doc_id = self._get_document_id(result)
                combined_scores[doc_id] = self.semantic_weight * result.get("score", 0)
                result_data[doc_id] = result
                result_data[doc_id]["search_type"] = "semantic"
            
            # Process keyword results
            for result in keyword_results:
                doc_id = self._get_document_id(result)
                if doc_id in combined_scores:
                    # Document found in both searches
                    combined_scores[doc_id] += self.keyword_weight * result.get("score", 0)
                    result_data[doc_id]["search_type"] = "hybrid"
                else:
                    # Document only in keyword search
                    combined_scores[doc_id] = self.keyword_weight * result.get("score", 0)
                    result_data[doc_id] = result
            
            # Sort by combined score
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Build final results
            final_results = []
            for doc_id, score in sorted_results[:limit]:
                if doc_id in result_data:
                    result = result_data[doc_id].copy()
                    result["score"] = score
                    result["hybrid_score"] = score
                    final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            return semantic_results[:limit]
    
    #get uniqe id document
    def _get_document_id(self, result: Dict[str, Any]) -> str:
        payload = result.get("payload", {})
        question = payload.get("question", "")
        # Use question text as ID (simplified approach)
        return question[:100]  # Truncate for consistency
    
    #preprocess text
    def _preprocess_text(self, text: str) -> str:
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation except apostrophes
            text = re.sub(r"[^\w\s']", " ", text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return text
    
    #Get search engine statistics
    def get_search_stats(self) -> Dict[str, Any]:
        return {
            "indexed_documents": len(self.documents),
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "tfidf_features": self.tfidf_vectorizer.max_features if self.tfidf_vectorizer else 0,
            "vectorizer_ready": self.tfidf_vectorizer is not None
        }