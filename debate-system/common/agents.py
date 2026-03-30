"""Agent implementations for the multi-agent system."""

from typing import List, Dict
from common.clients import OpenRouterClient
from common.prompts import get_dlb_system_prompt, get_pnm_system_prompt
from common.rag import RAGSystem
from common.utils import keep_recent_messages


class Agent:
    """Base class for debate agents."""

    def __init__(self, name: str, backend: OpenRouterClient, system_prompt: str, rag_system: RAGSystem):
        """
        Initialize an agent.

        Args:
            name: Agent name (e.g., "DLB", "PNM")
            backend: LLM client backend
            system_prompt: System prompt for the agent
            rag_system: RAG system for context retrieval
        """
        self.name = name
        self.backend = backend
        self.rag_system = rag_system
        self.messages = [{"role": "system", "content": system_prompt}]

    def generate(self, message: str, max_tokens: int = 150) -> str:
        """
        Generate a response to the given message.

        Args:
            message: Input message to respond to
            max_tokens: Maximum tokens for the response

        Returns:
            Generated response
        """
        # Retrieve relevant context
        context = self.rag_system.retrieve_context(message)

        # Keep only recent messages to manage context length
        self.messages = keep_recent_messages(self.messages)

        # Add user message with context
        self.messages.append({
            "role": "user",
            "content": f"[Context]\n{context}\n\n{message}"
        })

        # Generate response
        response = self.backend.chat(self.messages, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": response})

        return response


class DLBAgent(Agent):
    """Design and Link Budget Expert agent."""

    def __init__(self, backend: OpenRouterClient, rag_system: RAGSystem):
        """Initialize DLB agent."""
        super().__init__(
            name="DLB",
            backend=backend,
            system_prompt=get_dlb_system_prompt(),
            rag_system=rag_system
        )


class PNMAgent(Agent):
    """Payload and Network Management Expert agent."""

    def __init__(self, backend: OpenRouterClient, rag_system: RAGSystem):
        """Initialize PNM agent."""
        super().__init__(
            name="PNM",
            backend=backend,
            system_prompt=get_pnm_system_prompt(),
            rag_system=rag_system
        )
