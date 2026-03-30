"""OpenRouter API client wrapper."""

from typing import List, Dict, Optional
import os
from openai import OpenAI

from configs.settings import Config


class OpenRouterClient:
    """Client for interacting with LLM API (OpenRouter or similar)."""

    def __init__(self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            model: Model name to use
            api_key: API key. If None, will look for Config.API_KEY.
            api_base: API base URL. If None, will look for Config.API_BASE.
        """
        self.api_key = api_key or Config.API_KEY
        if not self.api_key:
            raise ValueError("API key not provided. Set API_KEY in .env or pass api_key parameter.")

        self.api_base = api_base or Config.API_BASE

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
        self.model = model

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.7) -> str:
        """
        Send chat completion request.

        Args:
            messages: List of messages with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response text
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            "model": self.model,
            "api_base": "https://openrouter.ai/api/v1"
        }