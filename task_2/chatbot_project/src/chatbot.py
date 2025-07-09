import logging
from typing import Dict, List, Any, Optional
import numpy as np
import time

from .config import Config
from .database import QdrantDatabase
from .llm import OpenAILLM
from .ollama_llm import OllamaLLM
from .embeddings import EmbeddingHandler
from .hybrid_search import HybridSearchEngine
from .security import SecurityHandler
from .quality_scorer import AdvancedQualityScorer

#Import Pinecone
try:
    from .pinecone_database import PineconeDatabase
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

logger = logging.getLogger(__name__)

#chatbot with dual database, dual LLM, and hybrid search
class NawatechChatbot:
    def __init__(self, config: Config):
        self.config = config
        self.primary_db = None
        self.secondary_db = None
        self.openai_llm = None
        self.ollama_llm = None
        self.current_llm = None
        self.embeddings = None
        self.hybrid_search = None
        self.security = None
        self.quality_scorer = None
        self.is_initialized = False
        self.conversation_history = []
        
        self.active_database = "primary"
        self.active_llm_type = config.LLM_TYPE
        self.active_search_type = config.SEARCH_TYPE
        
        self._initialize_components()
    
    #Initialize all chatbot components
    def _initialize_components(self):
        try:
            logger.info("Initializing advanced chatbot components...")
            
            #Initialize embedding handler
            self.embeddings = EmbeddingHandler(self.config.EMBEDDING_MODEL)
            
            #Initialize databases
            self._initialize_databases()
            
            #Initialize LLMs
            self._initialize_llms()
            
            #Initialize hybrid search
            if self.config.SEARCH_TYPE in ["hybrid", "keyword"]:
                self.hybrid_search = HybridSearchEngine(
                    semantic_weight=self.config.SEMANTIC_WEIGHT,
                    keyword_weight=self.config.KEYWORD_WEIGHT
                )
            
            #Initialize security handler
            self.security = SecurityHandler(
                rate_limit_requests=self.config.RATE_LIMIT_REQUESTS,
                rate_limit_window=self.config.RATE_LIMIT_WINDOW,
                max_input_length=self.config.MAX_INPUT_LENGTH
            )
            
            #Initialize quality scorer
            self.quality_scorer = AdvancedQualityScorer()
            
            self.is_initialized = True
            logger.info("Advanced chatbot components initialized success")
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot components: {e}")
            raise
    
    #Initialize database connections
    def _initialize_databases(self):
        try:
            # Primary database (Qdrant)
            self.primary_db = QdrantDatabase(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT,
                collection_name=self.config.QDRANT_COLLECTION_NAME
            )
            
            # Secondary database (Pinecone)
            if (PINECONE_AVAILABLE and hasattr(self.config, 'PINECONE_API_KEY') and 
                self.config.PINECONE_API_KEY):
                try:
                    self.secondary_db = PineconeDatabase(
                        api_key=self.config.PINECONE_API_KEY,
                        index_name=self.config.PINECONE_INDEX_NAME,
                        environment=self.config.PINECONE_ENVIRONMENT
                    )
                    logger.info("Dual database mode: Qdrant + Pinecone")
                except Exception as e:
                    logger.warning(f"Failed to initialize Pinecone: {e}")
                    self.secondary_db = None
            else:
                logger.info("Single database mode: Qdrant only")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    #Initialize LLM
    def _initialize_llms(self):
        try:
            #OpenAI
            if self.config.OPENAI_API_KEY:
                self.openai_llm = OpenAILLM(
                    api_key=self.config.OPENAI_API_KEY,
                    model=self.config.OPENAI_MODEL,
                    temperature=self.config.OPENAI_TEMPERATURE,
                    max_tokens=self.config.OPENAI_MAX_TOKENS
                )
            
            #Ollama
            self.ollama_llm = OllamaLLM(
                host=self.config.OLLAMA_HOST,
                model=self.config.OLLAMA_MODEL
            )
            
            #Set current LLM
            self.current_llm = self.openai_llm if self.active_llm_type == "openai" else self.ollama_llm
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise
    
    #Setup databases with FAQ data
    def setup_database(self, csv_path: str):
        try:
            logger.info("Setting up databases with FAQ data...")
            
            #Load FAQ data
            documents = self.primary_db.load_faq_data(csv_path)
            
            if not documents:
                raise ValueError("No FAQ data found")
            
            #Create embeddings
            texts = [doc["question"] for doc in documents]
            embeddings = self.embeddings.encode(texts)
            
            #Setup primary database
            self.primary_db.create_collection(self.embeddings.get_embedding_dimension())
            self.primary_db.add_documents(documents, embeddings)
            
            #Setup secondary database
            if self.secondary_db:
                try:
                    self.secondary_db.add_documents(documents, embeddings)
                    logger.info("Documents added to both databases")
                except Exception as e:
                    logger.warning(f"Failed to add documents to secondary database: {e}")
            
            #Setup hybrid search
            if self.hybrid_search:
                self.hybrid_search.index_documents(documents)
                logger.info("Hybrid search engine initialized")
            
            logger.info(f"Database setup completed with {len(documents)} FAQ items")
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
    
    #Process user query
    def chat(self, query: str, client_id: str = "default") -> Dict[str, Any]:
        try:
            if not self.is_initialized:
                raise ValueError("Chatbot not initialized")
            
            logger.info(f"Processing query with {self.active_llm_type} LLM and {self.active_database} database")
            
            #Security validation
            is_valid, security_message, security_info = self.security.validate_request(query, client_id)
            if not is_valid:
                return {
                    "response": security_message,
                    "overall_score": 0.0,
                    "confidence": 0.0,
                    "security_status": "blocked",
                    "security_info": security_info,
                    "active_llm": self.active_llm_type,
                    "active_database": self.active_database,
                    "active_search": self.active_search_type
                }
            
            #Use sanitized input
            sanitized_query = security_info.get("sanitized_input", query)
            
            #Generate query embedding
            query_embedding = self.embeddings.encode(sanitized_query.strip())
            if query_embedding.ndim > 1:
                query_embedding = query_embedding[0]
            
            #Search with current database
            context = self._search_documents(query_embedding, sanitized_query)
            
            #Generate response with current LLM
            response_data = self.llm.generate_response(
                query=sanitized_query,
                context=context,
                conversation_history=self.conversation_history
            )

            
            #Advanced quality scoring
            quality_metrics = self.quality_scorer.calculate_comprehensive_score(
                query=sanitized_query,
                response=response_data["response"],
                context=context,
                response_metadata=response_data
            )
            
            #Update conversation history
            self._update_conversation_history(sanitized_query, response_data["response"])
            
            #Combine metadata
            comprehensive_response = {
                **response_data,
                **quality_metrics,
                "security_status": "passed",
                "security_info": security_info,
                "active_llm": self.active_llm_type,
                "active_database": self.active_database,
                "active_search": self.active_search_type,
                "retrieval_results": len(context),
                "top_similarity_score": context[0]["score"] if context else 0.0,
                "timestamp": time.time()
            }
            
            logger.info("Advanced query processed successfully")
            return comprehensive_response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "response": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "overall_score": 0.0,
                "confidence": 0.0,
                "error": str(e),
                "active_llm": self.active_llm_type,
                "active_database": self.active_database,
                "timestamp": time.time()
            }
    
    #Search documents
    def _search_documents(self, query_embedding: np.ndarray, query: str) -> List[Dict[str, Any]]:
        try:
            #Select active database
            active_db = self.primary_db if self.active_database == "primary" else self.secondary_db
            if not active_db:
                active_db = self.primary_db
            
            #Perform search
            context = active_db.search(
                query_embedding=query_embedding,
                limit=self.config.TOP_K_RESULTS,
                score_threshold=self.config.SIMILARITY_THRESHOLD
            )
            
            #Apply hybrid search
            if self.hybrid_search and self.active_search_type in ["hybrid", "keyword"]:
                context = self.hybrid_search.search(
                    query=query,
                    semantic_results=context,
                    limit=self.config.TOP_K_RESULTS
                )
            
            return context
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    #Switch between primary and secondary database
    def switch_database(self, use_secondary: bool = False) -> Dict[str, Any]:
        """Switch between primary and secondary database"""
        try:
            if use_secondary and self.secondary_db:
                self.primary_db, self.secondary_db = self.secondary_db, self.primary_db
                return {
                    "success": True,
                    "message": "Switched to secondary database (Pinecone)",
                    "active_db": "pinecone"
                }
            else:
                return {
                    "success": False,
                    "message": "Secondary database not available or already using primary",
                    "active_db": "qdrant"
                }
        except Exception as e:
            logger.error(f"Database switch failed: {e}")
            return {
                "success": False,
                "message": f"Switch failed: {e}",
                "active_db": "qdrant"
            }

    #Switch LLM
    def switch_llm(self, llm_type: str) -> Dict[str, Any]:
        try:
            if llm_type == "openai" and self.openai_llm:
                self.active_llm_type = "openai"
                self.current_llm = self.openai_llm
                return {"success": True, "message": "Switched to OpenAI LLM", "active_llm": "openai"}
            elif llm_type == "ollama" and self.ollama_llm:
                if self.ollama_llm.test_connection():
                    self.active_llm_type = "ollama"
                    self.current_llm = self.ollama_llm
                    return {"success": True, "message": "Switched to Ollama LLM", "active_llm": "ollama"}
                else:
                    return {"success": False, "message": "Ollama server not available", "active_llm": self.active_llm_type}
            else:
                return {"success": False, "message": "LLM not available", "active_llm": self.active_llm_type}
        except Exception as e:
            return {"success": False, "message": f"Switch failed: {e}", "active_llm": self.active_llm_type}
    
    #Switch search
    def switch_search_type(self, search_type: str) -> Dict[str, Any]:
        try:
            if search_type in ["semantic", "hybrid", "keyword"]:
                self.active_search_type = search_type
                return {"success": True, "message": f"Switched to {search_type} search", "active_search": search_type}
            else:
                return {"success": False, "message": "Invalid search type", "active_search": self.active_search_type}
        except Exception as e:
            return {"success": False, "message": f"Switch failed: {e}", "active_search": self.active_search_type}
    
    #get system information
    def get_system_info(self) -> Dict[str, Any]:
        try:
            info = {
                "status": "healthy" if self.is_initialized else "error",
                "active_settings": {
                    "database": self.active_database,
                    "llm": self.active_llm_type,
                    "search": self.active_search_type
                },
                "components": {
                    "primary_database": "connected" if self.primary_db else "error",
                    "secondary_database": "connected" if self.secondary_db else "not_configured",
                    "openai_llm": "connected" if self.openai_llm else "not_configured",
                    "ollama_llm": "connected" if self.ollama_llm and self.ollama_llm.test_connection() else "not_available",
                    "embeddings": "loaded" if self.embeddings else "error",
                    "hybrid_search": "enabled" if self.hybrid_search else "disabled",
                    "security": "enabled" if self.security else "disabled",
                    "quality_scorer": "enabled" if self.quality_scorer else "disabled"
                },
                "capabilities": {
                    "dual_database": self.secondary_db is not None,
                    "dual_llm": self.openai_llm is not None and self.ollama_llm is not None,
                    "hybrid_search": self.hybrid_search is not None,
                    "security_protection": self.security is not None,
                    "advanced_quality": self.quality_scorer is not None
                },
                "config": {
                    "model": self.config.OPENAI_MODEL,
                    "embedding_model": self.config.EMBEDDING_MODEL,
                    "search_type": self.config.SEARCH_TYPE,
                    "database_type": "dual" if self.secondary_db else "single",
                    "security_enabled": self.security is not None,
                    "advanced_scoring": self.quality_scorer is not None
                }

            }
            
            #Add database info
            if self.primary_db:
                info["primary_database_info"] = self.primary_db.get_collection_info()
            
            if self.secondary_db:
                info["secondary_database_info"] = self.secondary_db.get_collection_info()
            
            #Add security stats
            if self.security:
                info["security_stats"] = self.security.get_security_stats()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"status": "error", "error": str(e)}
    
    #Update conversation history
    def _update_conversation_history(self, query: str, response: str):
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    #Reset conversation history
    def reset_conversation(self):
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    #Pertanyaan bantuan
    def get_suggested_questions(self) -> List[str]:
        return [
            "Apa itu Nawatech?",
            "Siapa CEO Nawatech?",
            "Bagaimana cara menghubungi Nawatech?",
            "Apa layanan yang ditawarkan Nawatech?",
            "Bagaimana cara Nawatech membantu bisnis?",
            "Teknologi apa yang digunakan Nawatech?",
            "Apakah Nawatech memiliki kantor?",
            "Apa visi Nawatech?",
            "Bagaimana cara bergabung dengan Nawatech?"
        ]
    
    #get scurity
    def get_client_security_info(self, client_id: str = "default") -> Dict[str, Any]:
        if self.security:
            return self.security.get_client_stats(client_id)
        return {"error": "Security not enabled"}
    
    def benchmark_search_methods(self, query: str) -> Dict[str, Any]:
        try:
            query_embedding = self.embeddings.encode(query.strip())
            if query_embedding.ndim > 1:
                query_embedding = query_embedding[0]
            
            results = {}
            
            #Benchmark semantic search
            start_time = time.time()
            semantic_results = self.primary_db.search(
                query_embedding=query_embedding,
                limit=self.config.TOP_K_RESULTS,
                score_threshold=self.config.SIMILARITY_THRESHOLD
            )
            semantic_time = time.time() - start_time
            
            results["semantic"] = {
                "results_count": len(semantic_results),
                "avg_score": np.mean([r["score"] for r in semantic_results]) if semantic_results else 0,
                "response_time": semantic_time
            }
            
            #Benchmark hybrid search if available
            if self.hybrid_search:
                start_time = time.time()
                hybrid_results = self.hybrid_search.search(
                    query=query,
                    semantic_results=semantic_results,
                    limit=self.config.TOP_K_RESULTS
                )
                hybrid_time = time.time() - start_time
                
                results["hybrid"] = {
                    "results_count": len(hybrid_results),
                    "avg_score": np.mean([r["score"] for r in hybrid_results]) if hybrid_results else 0,
                    "response_time": hybrid_time
                }
            
            #Benchmark secondary database if available
            if self.secondary_db:
                start_time = time.time()
                secondary_results = self.secondary_db.search(
                    query_embedding=query_embedding,
                    limit=self.config.TOP_K_RESULTS,
                    score_threshold=self.config.SIMILARITY_THRESHOLD
                )
                secondary_time = time.time() - start_time
                
                results["secondary_db"] = {
                    "results_count": len(secondary_results),
                    "avg_score": np.mean([r["score"] for r in secondary_results]) if secondary_results else 0,
                    "response_time": secondary_time
                }
            
            return {"benchmark_results": results, "timestamp": time.time()}
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"error": str(e)}
    
    #Switch OpenAI and Ollama
    def switch_llm_model(self, model_name: str, ollama_model: str = None):
        if model_name == "OpenAI":
            self.llm = OpenAILLM(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.OPENAI_MODEL,
                temperature=self.config.OPENAI_TEMPERATURE,
                max_tokens=self.config.OPENAI_MAX_TOKENS
            )
            logger.info("Switched to OpenAI model")
        elif model_name == "Ollama":
            from .ollama_llm import OllamaLLM
            ollama = OllamaLLM(
                host=getattr(self.config, 'OLLAMA_HOST', 'http://localhost:11434'),
                model=ollama_model or getattr(self.config, 'OLLAMA_MODEL', 'llama3.2')
            )
            self.llm = ollama
            logger.info(f"Switched to Ollama model: {ollama.model}")
        else:
            raise ValueError("Unsupported LLM model")


