"""
API Rotator Module
==================
Handles rotation of Google Gemini API keys to avoid quota/rate limits.
"""
import os
from google import genai
from typing import List, Any, Callable

class APIRotator:
    """Rotates through a list of Google Gemini API keys until a successful API call is made."""
    
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        self.model_name = model_name
        self.api_keys = self._load_api_keys()
        self._current_index = 0
        self.available = len(self.api_keys) > 0
        
    def _load_api_keys(self) -> List[str]:
        """Load and clean API keys from environment variables."""
        keys = []
        
        # Parse comma-separated keys from AI_GEN_API_KEYS
        keys_str = os.getenv('AI_GEN_API_KEYS', '')
        if keys_str:
            keys_str = keys_str.strip('"').strip("'")
            for k in keys_str.split(','):
                stripped = k.strip()
                if stripped:
                    keys.append(stripped)
        
        # Add backup key from GOOGLE_API_KEY
        single_key = os.getenv('GOOGLE_API_KEY', '').strip()
        if single_key:
            keys.append(single_key)
            
        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique_keys.append(k)
                
        return unique_keys

    def execute(self, api_func: Callable[[genai.Client], Any]) -> Any:
        """
        Execute an API call function, rotating keys if it fails.
        
        Args:
            api_func: A function/callable that accepts a genai.Client and returns a result.
            
        Returns:
            The return value of api_func on success.
            
        Raises:
            ConnectionError: If no API keys are configured.
            Exception: The last encountered exception if all keys fail.
        """
        if not self.api_keys:
            raise ConnectionError("No API keys configured. Please add AI_GEN_API_KEYS or GOOGLE_API_KEY in your .env file.")
            
        last_error = None
        num_keys = len(self.api_keys)
        
        for attempt in range(num_keys):
            index = (self._current_index + attempt) % num_keys
            try:
                # Create a client for this key
                client = genai.Client(api_key=self.api_keys[index])
                
                # Execute the provided function
                result = api_func(client)
                
                # If successful, remember this key index for the next call
                self._current_index = index
                self.available = True
                return result
            except Exception as e:
                last_error = e
                continue
                
        # If we reach here, all keys failed
        self.available = False
        raise last_error

    def generate_content(self, prompt: str) -> str:
        """Generate content by rotating keys until one works."""
        def call_gemini(client):
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
            
        return self.execute(call_gemini)
