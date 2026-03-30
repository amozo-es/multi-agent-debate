"""Moderator implementation for the multi-agent system."""

from typing import List, Dict
import time

from common.clients import OpenRouterClient
from common.prompts import (
    get_moderator_system_prompt,
    get_moderator_analyze_instructions,
    get_moderator_final_decision_instructions
)
from common.rag import RAGSystem
from common.utils import keep_recent_messages
from configs.settings import Config


class Moderator:
    """Moderator for managing debates between agents."""

    def __init__(self, topic: str, backend: OpenRouterClient, rag_system: RAGSystem,
                 max_rounds: int = None, max_messages: int = None):
        """
        Initialize the moderator.

        Args:
            topic: Debate topic
            backend: LLM client backend
            rag_system: RAG system for context retrieval
            max_rounds: Maximum number of debate rounds
            max_messages: Maximum total messages per agent
        """
        self.topic = topic
        self.backend = backend
        self.rag_system = rag_system
        self.max_rounds = max_rounds or Config.MAX_ROUNDS
        self.max_messages = max_messages or Config.MAX_MESSAGES

        self.round = 1
        self.total_messages = 0
        self.messages = [{"role": "system", "content": get_moderator_system_prompt(topic)}]

    def initiate_debate(self, max_tokens: int = 150) -> str:
        """
        Initiate the debate with an opening statement.

        Args:
            max_tokens: Maximum tokens for the response

        Returns:
            Opening statement
        """
        # Retrieve context for the topic
        context = self.rag_system.retrieve_context(self.topic)

        # Add user message
        self.messages.append({
            "role": "user",
            "content": f"[Context]\n{context}\n\nDebate topic: {self.topic}"
        })

        # Generate response
        response = self.backend.chat(self.messages, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": response})

        return response

    def analyze_and_respond(self, msg1: str, msg2: str, max_tokens: int = 150) -> str:
        """
        Analyze agent responses and provide new instructions.

        Args:
            msg1: Message from agent 1 (DLB)
            msg2: Message from agent 2 (PNM)
            max_tokens: Maximum tokens for the response

        Returns:
            Moderator's analysis and new instructions
        """
        # Get appropriate instructions based on round
        instructions = get_moderator_analyze_instructions(self.round)

        # Retrieve additional context based on agent messages
        extra_context = self.rag_system.retrieve_context(f"{msg1} {msg2}")
        instructions += f"\n[Context]{extra_context}"

        # Keep only recent messages
        self.messages = keep_recent_messages(self.messages)

        # Add user message with instructions and agent responses
        self.messages.append({
            "role": "user",
            "content": f"{instructions}\n\nDLB said: {msg1}\nPNM said: {msg2}"
        })

        # Generate response
        response = self.backend.chat(self.messages, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": response})

        return response

    def final_decision(self, max_tokens: int = 150) -> str:
        """
        Generate the final decision based on the entire debate.

        Args:
            max_tokens: Maximum tokens for the response

        Returns:
            Final technical decision
        """
        # Get final decision instructions
        instructions = get_moderator_final_decision_instructions()

        # Add topic context
        payload = f"Debate topic: {self.topic}"

        # Keep only recent messages
        self.messages = keep_recent_messages(self.messages)

        # Add user message
        self.messages.append({
            "role": "user",
            "content": f"{instructions}\n\n{payload}"
        })

        # Generate response
        response = self.backend.chat(self.messages, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": response})

        return response

    def check_termination(self) -> bool:
        """
        Check if the debate should terminate.

        Returns:
            True if debate should end, False otherwise
        """
        self.round += 1
        self.total_messages += 2
        return self.round > self.max_rounds or self.total_messages >= self.max_messages
