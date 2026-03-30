#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Homogeneous multi-agent system entry point."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.clients import OpenRouterClient
from common.rag import RAGSystem
from common.utils import slugify
from configs.settings import Config
from common.agents import DLBAgent, PNMAgent
from common.moderator import Moderator
from common.engine import run_debate_session


def main():
    """Main function for the homogeneous multi-agent system."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Homogeneous multi-agent system")
    parser.add_argument("--topic", required=True, help="Debate topic")
    parser.add_argument("--outfile", default=None, help="Output file path (optional)")
    parser.add_argument("--decision_dir", default=None, help="Directory for final decision JSON")
    parser.add_argument("--model", default=None, help="Model name (uses default if not provided)")
    args = parser.parse_args()

    # Validate configuration
    Config.validate()

    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = RAGSystem()

    # Initialize model backend (Single backend for homogeneous)
    model_name = args.model or Config.DEFAULT_HOMOGENEOUS_MODEL
    backend = OpenRouterClient(model_name)

    # Initialize participants
    moderator = Moderator(args.topic, backend, rag_system)
    dlb = DLBAgent(backend, rag_system)
    pnm = PNMAgent(backend, rag_system)

    # Setup output
    outpath = Path(args.outfile) if args.outfile else Path(f"output_{slugify(args.topic)}.txt")
    decision_dir = Path(args.decision_dir).resolve() if args.decision_dir else None

    # Run debate session
    result = run_debate_session(
        topic=args.topic,
        moderator=moderator,
        dlb=dlb,
        pnm=pnm,
        outpath=outpath,
        decision_dir=decision_dir
    )

    print(f"\nFull debate transcript saved to: {outpath}")


if __name__ == "__main__":
    main()