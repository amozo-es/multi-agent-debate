#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Individual LLM system with RAG enhancement."""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.clients import OpenRouterClient
from common.prompts import get_rag_agent_system_prompt
from common.rag import RAGSystem
from common.utils import slugify
from configs.settings import Config


class IndividualAgent:
    """Individual RAG-enhanced agent for answering technical questions."""

    def __init__(self, model_name: str, rag_system: RAGSystem):
        """
        Initialize the individual agent.

        Args:
            model_name: Name of the LLM model to use
            rag_system: RAG system for context retrieval
        """
        self.backend = OpenRouterClient(model_name)
        self.rag_system = rag_system
        self.messages = [{"role": "system", "content": get_rag_agent_system_prompt()}]

    def generate_response(self, topic: str, max_tokens: int = 150) -> str:
        """
        Generate a response for the given topic.

        Args:
            topic: Question or topic to address
            max_tokens: Maximum tokens for the response

        Returns:
            Generated response
        """
        # Retrieve relevant context
        context = self.rag_system.retrieve_context(topic)

        # Add user message with context
        self.messages.append({
            "role": "user",
            "content": f"[Context]\n{context}\n\n{topic}"
        })

        # Generate response
        response = self.backend.chat(self.messages, max_tokens=max_tokens)
        self.messages.append({"role": "assistant", "content": response})

        return response


def main():
    """Main function for the individual LLM system."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Individual LLM system with RAG")
    parser.add_argument("--topic", required=True, help="Question or topic to address")
    parser.add_argument("--outfile", default=None, help="Output file path (optional)")
    parser.add_argument("--model", default=None, help="Model name (uses default if not provided)")
    args = parser.parse_args()

    # Validate configuration
    Config.validate()

    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = RAGSystem()

    # Initialize agent
    model_name = args.model or Config.DEFAULT_INDIVIDUAL_MODEL
    agent = IndividualAgent(model_name, rag_system)

    # Generate response
    print(f"Processing topic: {args.topic}")
    response = agent.generate_response(args.topic)

    # Prepare output
    result = {
        "topic": args.topic,
        "final_decision": response
    }

    # Save output
    if args.outfile:
        outpath = Path(args.outfile).resolve()
    else:
        outpath = Path(f"output_{slugify(args.topic)}.json")

    outpath.parent.mkdir(parents=True, exist_ok=True)

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Response saved to: {outpath}")
    print("\nGenerated response:")
    print(response)


if __name__ == "__main__":
    main()