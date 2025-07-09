# Format .env
# OpenAI Configuration
OPENAI_API_KEY=xxx

# Database Configuration
DATABASE_TYPE=qdrant  # qdrant or pinecone

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Pinecone Configuration
PINECONE_API_KEY=xxx
PINECONE_INDEX_NAME=nawatech-faq
PINECONE_ENVIRONMENT=gcp-starter

# LLM Configuration
LLM_TYPE=openai  # openai or ollama

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Search Configuration
SEARCH_TYPE=hybrid  # semantic, keyword, or hybrid

# Security Configuration
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60
MAX_INPUT_LENGTH=500

