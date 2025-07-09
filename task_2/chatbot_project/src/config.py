import os
from dotenv import load_dotenv

load_dotenv()

#Load API key from Docker secret file
def load_api_key_from_secret(secret_path):
    try:
        with open(secret_path) as f:
            return f.read().strip()
    except Exception:
        return None

#config class
class Config:
    # API Keys (support Docker Secrets & .env)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or load_api_key_from_secret(os.getenv("OPENAI_API_KEY_FILE"))
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or load_api_key_from_secret(os.getenv("PINECONE_API_KEY_FILE"))
    
    #Database Settings
    DATABASE_TYPE = os.getenv("DATABASE_TYPE", "qdrant")  # qdrant or pinecone
    
    #Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME = "nawatech_faq"
    
    #Pinecone
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nawatech-faq")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    
    #LLM Settings
    LLM_TYPE = os.getenv("LLM_TYPE", "openai")  # openai or ollama
    
    #OpenAI Settings
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE = 0.1
    OPENAI_MAX_TOKENS = 500
    
    #Ollama Settings
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    
    #Embedding Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    #Search Settings
    SEARCH_TYPE = os.getenv("SEARCH_TYPE", "hybrid")  # semantic, keyword, or hybrid
    TOP_K_RESULTS = 3
    SIMILARITY_THRESHOLD = 0.7
    KEYWORD_WEIGHT = 0.3  # Weight for keyword search in hybrid mode
    SEMANTIC_WEIGHT = 0.7  # Weight for semantic search in hybrid mode
    
    #Security Settings
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 60))  # requests per minute
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))    # seconds
    MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 500))     # characters
    
    #Streamlit
    PAGE_TITLE = "Nawatech FAQ Chatbot"
    PAGE_ICON = "🤖"
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY and cls.LLM_TYPE == "openai":
            raise ValueError("OPENAI_API_KEY is required when using OpenAI")
        
        if not cls.PINECONE_API_KEY and cls.DATABASE_TYPE == "pinecone":
            raise ValueError("PINECONE_API_KEY is required when using Pinecone")
        
        return True