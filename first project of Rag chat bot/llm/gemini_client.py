"""
Google Gemini API Client for PDF Tutor AI.
Uses the official Google Gen AI Python SDK (google-genai).
Completely optional: the application runs fully in Retrieval-Only mode
when no GEMINI_API_KEY is available.
"""

import os
from typing import Optional
from google import genai
from google.genai import errors

from config.settings import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY_ENV
from llm.prompts import TUTOR_SYSTEM_PROMPT


class GeminiAPIError(Exception):
    """Raised when an error occurs during Gemini API generation."""
    pass


class GeminiClient:
    """
    Encapsulates interaction with the Google Gemini API.
    Designed for zero local neural inference.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv(GEMINI_API_KEY_ENV, "").strip()
        self._client: Optional[genai.Client] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Google Gen AI client if an API key is available."""
        if self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None
        else:
            self._client = None

    def update_api_key(self, api_key: str) -> None:
        """Update API key at runtime."""
        self.api_key = api_key.strip()
        self._initialize_client()

    def set_model(self, model_name: str) -> None:
        """Update model name."""
        self.model_name = model_name

    def is_available(self) -> bool:
        """Returns True if the client has a configured API key."""
        return self._client is not None and bool(self.api_key)

    def generate_tutoring_response(
        self,
        prompt: str,
        system_instruction: str = TUTOR_SYSTEM_PROMPT,
        temperature: float = 0.4,
    ) -> str:
        """
        Generate educational response using Gemini.
        
        Args:
            prompt: Structured prompt containing retrieved passages and query.
            system_instruction: The 13-rule educational tutor system prompt.
            temperature: Creativity control (0.4 for grounded pedagogical answers).
            
        Returns:
            Generated natural language response with citations.
            
        Raises:
            GeminiAPIError: If generation fails or API key is invalid.
        """
        if not self.is_available():
            raise GeminiAPIError(
                "Gemini API key is not configured. The application is running in Retrieval-Only mode."
            )
            
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "temperature": temperature,
                },
            )
            
            if response and response.text:
                return response.text
            return "The tutor could not generate a response for this query. Please check your query."
        except errors.APIError as e:
            raise GeminiAPIError(f"Gemini API Error ({e.code}): {e.message}")
        except Exception as e:
            raise GeminiAPIError(f"Failed to communicate with Gemini API: {str(e)}")
