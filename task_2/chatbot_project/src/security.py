import time
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)

#
class SecurityHandler:
    def __init__(self, 
                 rate_limit_requests: int = 60, 
                 rate_limit_window: int = 60,
                 max_input_length: int = 500):

        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.max_input_length = max_input_length
        
        # Rate limiting storage
        self.request_history = defaultdict(deque)
        
        # Suspicious patterns for prompt injection detection
        self.injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"disregard\s+the\s+above",
            r"forget\s+everything",
            r"new\s+instructions",
            r"system\s*:\s*",
            r"assistant\s*:\s*",
            r"human\s*:\s*",
            r"ai\s*:\s*",
            r"</\s*instructions\s*>",
            r"<\s*instructions\s*>",
            r"act\s+as\s+if",
            r"pretend\s+to\s+be",
            r"role\s*play",
            r"jailbreak",
            r"bypass\s+filter",
            r"ignore\s+safety"
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns]
        
        # Blocked content patterns
        self.blocked_patterns = [
            r"hack\s+into",
            r"ddos\s+attack",
            r"sql\s+injection",
            r"malware",
            r"virus\s+code",
            r"exploit\s+vulnerability"
        ]
        
        self.compiled_blocked = [re.compile(pattern, re.IGNORECASE) for pattern in self.blocked_patterns]
    
    #Validate incoming request for security issues
    def validate_request(self, user_input: str, client_id: str = "default") -> Tuple[bool, str, Dict[str, Any]]:
        try:
            security_info = {
                "rate_limit_status": "ok",
                "input_validation_status": "ok",
                "injection_risk": "low",
                "blocked_content": False,
                "sanitized_input": user_input
            }
            
            # Check rate limiting
            rate_limit_ok, rate_message = self._check_rate_limit(client_id)
            if not rate_limit_ok:
                security_info["rate_limit_status"] = "exceeded"
                return False, rate_message, security_info
            
            # Check input length
            if len(user_input) > self.max_input_length:
                security_info["input_validation_status"] = "too_long"
                return False, f"Input too long. Maximum {self.max_input_length} characters allowed.", security_info
            
            # Check for empty/invalid input
            if not user_input or user_input.strip() == "":
                security_info["input_validation_status"] = "empty"
                return False, "Input cannot be empty.", security_info
            
            # Check for prompt injection attempts
            injection_risk, injection_message = self._detect_prompt_injection(user_input)
            security_info["injection_risk"] = injection_risk
            if injection_risk == "high":
                return False, injection_message, security_info
            
            # Check for blocked content
            blocked, blocked_message = self._check_blocked_content(user_input)
            security_info["blocked_content"] = blocked
            if blocked:
                return False, blocked_message, security_info
            
            # Sanitize input
            sanitized_input = self._sanitize_input(user_input)
            security_info["sanitized_input"] = sanitized_input
            
            return True, "Valid request", security_info
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return False, "Security validation error", {"error": str(e)}
    
    #Check if client has exceeded rate limit
    def _check_rate_limit(self, client_id: str) -> Tuple[bool, str]:
        try:
            current_time = time.time()
            client_requests = self.request_history[client_id]
            
            # Remove old requests outside the window
            while client_requests and current_time - client_requests[0] > self.rate_limit_window:
                client_requests.popleft()
            
            # Check if within limit
            if len(client_requests) >= self.rate_limit_requests:
                return False, f"Rate limit exceeded. Maximum {self.rate_limit_requests} requests per {self.rate_limit_window} seconds."
            
            # Add current request
            client_requests.append(current_time)
            
            return True, "Within rate limit"
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, "Rate limit check error"  # Allow on error to avoid blocking legitimate users
    
    #Detect potential prompt injection attempts
    def _detect_prompt_injection(self, user_input: str) -> Tuple[str, str]:
        try:
            injection_count = 0
            detected_patterns = []
            
            # Check against injection patterns
            for pattern in self.compiled_patterns:
                if pattern.search(user_input):
                    injection_count += 1
                    detected_patterns.append(pattern.pattern)
            
            # Determine risk level
            if injection_count >= 3:
                risk_level = "high"
                message = "Potential prompt injection detected. Please rephrase your question."
            elif injection_count >= 1:
                risk_level = "medium"
                message = f"Suspicious patterns detected: {', '.join(detected_patterns[:2])}"
            else:
                risk_level = "low"
                message = "No injection patterns detected"
            
            return risk_level, message
            
        except Exception as e:
            logger.error(f"Prompt injection detection failed: {e}")
            return "low", "Detection error"
    
    #Check for blocked content patterns
    def _check_blocked_content(self, user_input: str) -> Tuple[bool, str]:
        try:
            for pattern in self.compiled_blocked:
                if pattern.search(user_input):
                    return True, "Content contains blocked patterns. Please ask about Nawatech-related topics."

            return False, "Content is safe"
        
        except Exception as e:
            logger.error(f"Blocked content check failed: {e}")
            return False, "Blocked content check error"
    
    def get_security_stats(self) -> Dict[str, Any]:
        return {
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
            "max_input_length": self.max_input_length,
            "tracked_clients": len(self.request_history)
        }
    
    def _sanitize_input(self, user_input: str) -> str:
        sanitized = re.sub(r'[^\w\s.,!?]', '', user_input)  # Remove weird char
        sanitized = sanitized.strip()
        return sanitized
    
    def get_client_stats(self, client_id: str = "default") -> Dict[str, Any]:
        client_requests = self.request_history.get(client_id, [])
        return {
            "client_id": client_id,
            "requests_in_window": len(client_requests),
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window,
            "max_input_length": self.max_input_length
        }




