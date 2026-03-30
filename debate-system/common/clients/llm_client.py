"""Generic LLM client interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
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
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        pass


class LLMClientFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_client(client_type: str, **kwargs) -> LLMClient:
        """
        Create an LLM client of the specified type.

        Args:
            client_type: Type of client to create ("openrouter")
            **kwargs: Additional arguments for client initialization

        Returns:
            Configured LLM client instance
        """
        if client_type.lower() == "openrouter":
            from .openrouter_client import OpenRouterClient
            return OpenRouterClient(**kwargs)
        else:
            raise ValueError(f"Unknown client type: {client_type}")