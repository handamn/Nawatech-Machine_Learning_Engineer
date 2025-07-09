import requests
import json
import logging
from typing import List, Dict, Any, Optional
import time
import os

logger = logging.getLogger(__name__)

#Handler for Ollama
class OllamaLLM:
    def __init__(self, host: str = None, model: str = None):
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.host = self.host.rstrip('/')
        self.api_url = f"{self.host}/api"
    
    #generate response
    def generate_response(self, query: str, context: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        try:
            # Prepare context
            context_text = self._prepare_context(context)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(context_text)
            
            # Create full prompt
            full_prompt = f"{system_prompt}\n\nPertanyaan: {query}\n\nJawaban:"
            
            logger.info(f"Generating response with Ollama model: {self.model}")
            
            # Make API call to Ollama
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response
            assistant_response = result.get("response", "").strip()
            
            # Calculate basic quality metrics
            quality_score = self._calculate_quality_score(query, assistant_response, context)
            
            result_data = {
                "response": assistant_response,
                "model": self.model,
                "quality_score": quality_score,
                "context_used": len(context),
                "tokens_used": result.get("eval_count", 0),
                "confidence": self._calculate_confidence(context),
                "source_references": self._extract_source_references(context)
            }
            
            logger.info("Response generated successfully with Ollama")
            return result_data
            
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama server")
            return {
                "response": "Maaf, model lokal tidak tersedia. Pastikan Ollama server berjalan.",
                "model": self.model,
                "quality_score": 0.0,
                "context_used": 0,
                "tokens_used": 0,
                "confidence": 0.0,
                "source_references": [],
                "error": "Ollama server connection failed"
            }
        except Exception as e:
            logger.error(f"Failed to generate response with Ollama: {e}")
            return {
                "response": "Maaf, terjadi kesalahan pada model lokal. Silakan coba lagi.",
                "model": self.model,
                "quality_score": 0.0,
                "context_used": 0,
                "tokens_used": 0,
                "confidence": 0.0,
                "source_references": [],
                "error": str(e)
            }
    
    #prep context
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        if not context:
            return "Tidak ada informasi yang ditemukan dalam basis data FAQ."
        
        context_parts = []
        for i, ctx in enumerate(context, 1):
            payload = ctx.get("payload", {})
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            score = ctx.get("score", 0.0)
            
            context_parts.append(f"FAQ {i} (relevance: {score:.3f}):\nQ: {question}\nA: {answer}")
        
        return "\n\n".join(context_parts)
    
    #prompt
    def _create_system_prompt(self, context: str) -> str:
        """Create system prompt for the chatbot"""
        return f"""Anda adalah asisten AI untuk Nawatech (PT. Nawa Darsana Teknologi), perusahaan pengembangan perangkat lunak.

INSTRUKSI PENTING:
1. Jawab pertanyaan HANYA berdasarkan informasi FAQ yang disediakan
2. Gunakan bahasa Indonesia yang ramah dan profesional
3. Jika pertanyaan tidak terkait dengan informasi yang tersedia, katakan bahwa Anda tidak memiliki informasi tersebut
4. Jangan membuat informasi baru yang tidak ada dalam FAQ
5. Selalu rujuk ke informasi yang paling relevan

INFORMASI FAQ YANG TERSEDIA:
{context}

Berikan jawaban yang akurat, informatif, dan membantu berdasarkan informasi FAQ di atas."""
    
    #calculate quality score
    def _calculate_quality_score(self, query: str, response: str, context: List[Dict[str, Any]]) -> float:
        try:
            score = 0.0
            
            # Context relevance (0.4)
            if context:
                avg_context_score = sum(ctx.get("score", 0.0) for ctx in context) / len(context)
                score += avg_context_score * 0.4
            
            # Response length appropriateness (0.3)
            response_length = len(response.split())
            if 10 <= response_length <= 100:
                score += 0.3
            elif response_length > 100:
                score += 0.2
            else:
                score += 0.1
            
            # Contains key information (0.3)
            if any(keyword in response.lower() for keyword in ["nawatech", "perusahaan", "teknologi"]):
                score += 0.3
            
            return min(score, 1.0)
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {e}")
            return 0.5
    
    #calculate confidence
    def _calculate_confidence(self, context: List[Dict[str, Any]]) -> float:
        if not context:
            return 0.0
        
        # Use average of top context scores
        scores = [ctx.get("score", 0.0) for ctx in context]
        return sum(scores) / len(scores)
    
    def _extract_source_references(self, context: List[Dict[str, Any]]) -> List[str]:
        references = []
        for ctx in context:
            payload = ctx.get("payload", {})
            question = payload.get("question", "")
            if question:
                references.append(question[:100] + "..." if len(question) > 100 else question)
        return references
    
    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
    
    #list available model
    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    #download model
    def pull_model(self, model_name: str) -> bool:
        try:
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes timeout for model download
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False