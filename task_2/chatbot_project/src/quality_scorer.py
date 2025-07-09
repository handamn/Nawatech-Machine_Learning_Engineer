import numpy as np
import re
from typing import List, Dict, Any, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

#Advanced quality scoring system
class AdvancedQualityScorer: 
    def __init__(self):
        self.company_keywords = [
            "nawatech", "nawa", "teknologi", "perusahaan", "software", "pengembangan",
            "bisnis", "solusi", "aplikasi", "sistem", "digital", "arfan", "arlanda"
        ]
        
        self.quality_indicators = [
            "membantu", "solusi", "layanan", "produk", "teknologi", "inovasi",
            "profesional", "berkualitas", "terpercaya", "berpengalaman"
        ]
        
        self.negative_indicators = [
            "tidak tahu", "tidak ada", "maaf", "error", "gagal", "tidak bisa",
            "tidak tersedia", "tidak ditemukan", "belum", "sementara"
        ]
    
    #Calculate comprehensive quality score
    def calculate_comprehensive_score(self, 
                                    query: str,
                                    response: str,
                                    context: List[Dict[str, Any]],
                                    response_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            scores = {}
            
            # 1. Context Relevance Score (25%)
            scores["context_relevance"] = self._calculate_context_relevance(context)
            
            # 2. Response Coherence Score (20%)
            scores["coherence"] = self._calculate_coherence(response)
            
            # 3. Factuality Score (20%)
            scores["factuality"] = self._calculate_factuality(response, context)
            
            # 4. Completeness Score (15%)
            scores["completeness"] = self._calculate_completeness(query, response)
            
            # 5. Language Quality Score (10%)
            scores["language_quality"] = self._calculate_language_quality(response)
            
            # 6. Domain Relevance Score (10%)
            scores["domain_relevance"] = self._calculate_domain_relevance(response)
            
            # Calculate weighted overall score
            weights = {
                "context_relevance": 0.25,
                "coherence": 0.20,
                "factuality": 0.20,
                "completeness": 0.15,
                "language_quality": 0.10,
                "domain_relevance": 0.10
            }
            
            overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
            
            # Calculate confidence based on context and consistency
            confidence = self._calculate_confidence(context, scores)
            
            # Determine quality tier
            quality_tier = self._determine_quality_tier(overall_score)
            
            return {
                "overall_score": round(overall_score, 3),
                "confidence": round(confidence, 3),
                "quality_tier": quality_tier,
                "detailed_scores": {k: round(v, 3) for k, v in scores.items()},
                "weights": weights,
                "recommendations": self._generate_recommendations(scores),
                "score_explanation": self._generate_explanation(scores, overall_score)
            }
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return {
                "overall_score": 0.5,
                "confidence": 0.0,
                "quality_tier": "unknown",
                "detailed_scores": {},
                "error": str(e)
            }
    
    #Calculate relevance of retrieved context
    def _calculate_context_relevance(self, context: List[Dict[str, Any]]) -> float:
        try:
            if not context:
                return 0.0
            
            # Average of context scores with decay for lower-ranked results
            total_score = 0.0
            total_weight = 0.0
            
            for i, ctx in enumerate(context):
                score = ctx.get("score", 0.0)
                weight = 1.0 / (i + 1)  # Decay weight for lower ranks
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Context relevance calculation failed: {e}")
            return 0.5
    
    #Calculate response coherence and readability
    def _calculate_coherence(self, response: str) -> float:
        try:
            if not response or len(response.strip()) < 10:
                return 0.0
            
            score = 0.0
            
            # Check sentence structure
            sentences = re.split(r'[.!?]+', response)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            if len(valid_sentences) > 0:
                # Penalize very short or very long responses
                length_score = min(1.0, len(response) / 200.0)  # Optimal around 200 chars
                if len(response) > 500:
                    length_score *= 0.8  # Penalize very long responses
                
                score += length_score * 0.4
                
                # Check for proper sentence structure
                proper_sentences = sum(1 for s in valid_sentences if len(s.split()) >= 3)
                sentence_score = proper_sentences / len(valid_sentences)
                score += sentence_score * 0.3
                
                # Check for repetition
                words = response.lower().split()
                unique_words = set(words)
                repetition_score = len(unique_words) / len(words) if words else 0
                score += repetition_score * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {e}")
            return 0.5
    
    #Calculate factual accuracy based on context alignment
    def _calculate_factuality(self, response: str, context: List[Dict[str, Any]]) -> float:
        try:
            if not context or not response:
                return 0.5
            
            response_lower = response.lower()
            
            # Extract facts from context
            context_facts = []
            for ctx in context:
                payload = ctx.get("payload", {})
                answer = payload.get("answer", "").lower()
                if answer:
                    context_facts.append(answer)
            
            if not context_facts:
                return 0.5
            
            # Calculate alignment with context facts
            alignment_score = 0.0
            for fact in context_facts:
                # Simple keyword overlap
                fact_words = set(fact.split())
                response_words = set(response_lower.split())
                overlap = len(fact_words.intersection(response_words))
                if len(fact_words) > 0:
                    alignment_score += overlap / len(fact_words)
            
            # Average alignment across all facts
            factuality_score = alignment_score / len(context_facts) if context_facts else 0.5
            
            # Penalize if response contains contradictory information
            negative_count = sum(1 for neg in self.negative_indicators if neg in response_lower)
            if negative_count > 0:
                factuality_score *= 0.7
            
            return min(1.0, factuality_score)
            
        except Exception as e:
            logger.error(f"Factuality calculation failed: {e}")
            return 0.5
    
    #Calculate how completely the response addresses the query
    def _calculate_completeness(self, query: str, response: str) -> float:
        try:
            if not query or not response:
                return 0.0
            
            query_lower = query.lower()
            response_lower = response.lower()
            
            # Extract key question words
            question_words = ["apa", "siapa", "dimana", "kapan", "mengapa", "bagaimana", "berapa"]
            question_type = None
            for word in question_words:
                if word in query_lower:
                    question_type = word
                    break
            
            score = 0.5  # Base score
            
            # Check if response addresses the question type
            if question_type:
                if question_type == "apa" and any(word in response_lower for word in ["adalah", "merupakan", "yaitu"]):
                    score += 0.3
                elif question_type == "siapa" and any(word in response_lower for word in ["nama", "orang", "person"]):
                    score += 0.3
                elif question_type == "dimana" and any(word in response_lower for word in ["lokasi", "alamat", "tempat"]):
                    score += 0.3
            
            # Check for key query terms in response
            query_words = set(query_lower.split())
            response_words = set(response_lower.split())
            term_coverage = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
            score += term_coverage * 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Completeness calculation failed: {e}")
            return 0.5
    
    #Calculate language quality and professionalism
    def _calculate_language_quality(self, response: str) -> float:
        try:
            if not response:
                return 0.0
            
            score = 0.0
            
            # Check for proper Indonesian language patterns
            indonesian_indicators = ["adalah", "dengan", "untuk", "dari", "yang", "dapat", "akan"]
            indonesian_count = sum(1 for word in indonesian_indicators if word in response.lower())
            if indonesian_count > 0:
                score += 0.3
            
            # Check for professional language
            professional_count = sum(1 for word in self.quality_indicators if word in response.lower())
            if professional_count > 0:
                score += 0.3
            
            # Check for proper punctuation
            if re.search(r'[.!?]$', response.strip()):
                score += 0.2
            
            # Check for capitalization
            if response[0].isupper():
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Language quality calculation failed: {e}")
            return 0.5
    
    #Calculate relevance to Nawatech domain
    def _calculate_domain_relevance(self, response: str) -> float:
        try:
            if not response:
                return 0.0
            
            response_lower = response.lower()
            
            # Count company-related keywords
            company_count = sum(1 for keyword in self.company_keywords if keyword in response_lower)
            
            # Normalize by response length
            word_count = len(response_lower.split())
            if word_count == 0:
                return 0.0
            
            # Calculate relevance score
            relevance_score = min(1.0, company_count / max(1, word_count / 10))
            
            return relevance_score
            
        except Exception as e:
            logger.error(f"Domain relevance calculation failed: {e}")
            return 0.5
    
    #calculate confidence
    def _calculate_confidence(self, context: List[Dict[str, Any]], scores: Dict[str, float]) -> float:
        try:
            # Base confidence from context quality
            context_confidence = self._calculate_context_relevance(context)
            
            # Consistency across metrics
            score_values = list(scores.values())
            if score_values:
                score_std = np.std(score_values)
                consistency = max(0, 1 - score_std)  # Lower std = higher consistency
            else:
                consistency = 0.5
            
            # Combine factors
            confidence = (context_confidence * 0.6 + consistency * 0.4)
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    #Determine quality tier
    def _determine_quality_tier(self, overall_score: float) -> str:
        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.6:
            return "good"
        elif overall_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    #Generate recommendations for improvement
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        recommendations = []
        
        if scores.get("context_relevance", 0) < 0.5:
            recommendations.append("Improve context retrieval or increase similarity threshold")
        
        if scores.get("coherence", 0) < 0.5:
            recommendations.append("Enhance response structure and readability")
        
        if scores.get("factuality", 0) < 0.5:
            recommendations.append("Ensure better alignment with source material")
        
        if scores.get("completeness", 0) < 0.5:
            recommendations.append("Address all aspects of the user query")
        
        if scores.get("domain_relevance", 0) < 0.5:
            recommendations.append("Include more company-specific information")
        
        return recommendations
    
    #Generate explanation of the quality score
    def _generate_explanation(self, scores: Dict[str, float], overall_score: float) -> str:
        tier = self._determine_quality_tier(overall_score)
        
        explanation = f"Overall quality is {tier} ({overall_score:.2f}). "
        
        # Highlight best and worst aspects
        best_aspect = max(scores.items(), key=lambda x: x[1])
        worst_aspect = min(scores.items(), key=lambda x: x[1])
        
        explanation += f"Strongest aspect: {best_aspect[0]} ({best_aspect[1]:.2f}). "
        explanation += f"Needs improvement: {worst_aspect[0]} ({worst_aspect[1]:.2f})."
        
        return explanation