#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Heterogeneous multi-agent system entry point."""

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
    """Main function for the heterogeneous multi-agent system."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Heterogeneous multi-agent system")
    parser.add_argument("--topic", required=True, help="Debate topic")
    parser.add_argument("--outfile", default=None, help="Output file path (optional)")
    parser.add_argument("--decision_dir", default=None, help="Directory for final decision JSON")
    
    # Model arguments
    parser.add_argument("--moderator_model", default=None, help="Model for the Moderator")
    parser.add_argument("--dlb_model", default=None, help="Model for the DLB Agent")
    parser.add_argument("--pnm_model", default=None, help="Model for the PNM Agent")
    
    args = parser.parse_args()

    # Validate configuration
    Config.validate()

    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = RAGSystem()

    # Initialize model backends (Three separate backends for heterogeneous)
    mod_model = args.moderator_model or Config.HETEROGENEOUS_MODERATOR_MODEL
    dlb_model = args.dlb_model or Config.HETEROGENEOUS_DLB_MODEL
    pnm_model = args.pnm_model or Config.HETEROGENEOUS_PNM_MODEL
    
    print(f"Moderator model: {mod_model}")
    print(f"DLB Agent model: {dlb_model}")
    print(f"PNM Agent model: {pnm_model}")

    backend_mod = OpenRouterClient(mod_model)
    backend_dlb = OpenRouterClient(dlb_model)
    backend_pnm = OpenRouterClient(pnm_model)

    # Initialize participants with their respective backends
    moderator = Moderator(args.topic, backend_mod, rag_system)
    dlb = DLBAgent(backend_dlb, rag_system)
    pnm = PNMAgent(backend_pnm, rag_system)

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
