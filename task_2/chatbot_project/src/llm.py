import openai
from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

#Handler for OpenAI LLM
class OpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.1, max_tokens: int = 500):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
    
    #generate response
    def generate_response(self, query: str, context: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        try:
            # Prepare context
            context_text = self._prepare_context(context)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(context_text)
            
            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-6:])  # Last 3 exchanges
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            logger.info(f"Generating response for query: {query[:50]}...")
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Extract response
            assistant_response = response.choices[0].message.content
            
            # Calculate basic quality metrics
            quality_score = self._calculate_quality_score(query, assistant_response, context)
            
            result = {
                "response": assistant_response,
                "model": self.model,
                "quality_score": quality_score,
                "context_used": len(context),
                "tokens_used": response.usage.total_tokens,
                "confidence": self._calculate_confidence(context),
                "source_references": self._extract_source_references(context)
            }
            
            logger.info("Response generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                "response": "Maaf, saya mengalami kesulitan dalam memproses pertanyaan Anda. Silakan coba lagi.",
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False