#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch processing script for running multiple debates."""

import sys
from pathlib import Path
import time
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.utils import slugify, run_with_retries, add_delay_between_requests
from configs.settings import Config


def iter_questions(txt_path: Path):
    """Read .txt file and yield non-empty questions (one per line)."""
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            yield q


def main():
    """Main function for batch processing debates."""
    if len(sys.argv) < 4:
        print("Usage: python launch_debates.py SCRIPT_PATH TXT_FOLDER OUTPUT_FOLDER [DECISION_FOLDER]")
        print("Example: python launch_debates.py main.py questions/ outputs/ decisions/")
        sys.exit(1)

    script_path = Path(sys.argv[1]).resolve()
    txt_dir = Path(sys.argv[2]).resolve()
    out_dir = Path(sys.argv[3]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_dir = None
    if len(sys.argv) >= 5:
        decision_dir = Path(sys.argv[4]).resolve()
        decision_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not script_path.exists():
        sys.exit(f"ERROR: Script not found: {script_path}")
    if not txt_dir.exists():
        sys.exit(f"ERROR: Questions folder not found: {txt_dir}")

    # Find all .txt files
    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        sys.exit(f"ERROR: No .txt files found in {txt_dir}")

    print(f"\n📂 Questions folder: {txt_dir}")
    print(f"📄 Output folder: {out_dir}")
    if decision_dir:
        print(f"🎯 Decision folder: {decision_dir}")
    print()

    # Process each question
    total_q = 0
    for txt_file in txt_files:
        print(f"--- Processing {txt_file.name} ---")

        for i, question in enumerate(iter_questions(txt_file), start=1):
            total_q += 1
            out_file = out_dir / f"{txt_file.stem}__q{i:02d}_{slugify(question)}.txt"

            print(f"\n🚀 [{total_q}] Launching question {i} from {txt_file.name}:")
            print(f"   {question[:100]}{'...' if len(question) > 100 else ''}")

            # Build command
            cmd = [
                sys.executable,
                str(script_path),
                "--topic", question,
                "--outfile", str(out_file)
            ]

            if decision_dir is not None:
                cmd += ["--decision_dir", str(decision_dir)]

            # Run with retries
            success = run_with_retries(
                cmd,
                max_retries=Config.MAX_RETRIES,
                base_delay=Config.RETRY_BASE_DELAY
            )

            if not success:
                print(f"❌ Failed to process question {i} from {txt_file.name}")
                continue

            # Add delay between requests
            add_delay_between_requests(Config.SLEEP_BASE, Config.SLEEP_JITTER)

    print(f"\n✅ Processed {total_q} questions total.")
    print(f"📁 Results saved in: {out_dir}")
    if decision_dir:
        print(f"🎯 Decisions saved in: {decision_dir}")


if __name__ == "__main__":
    main()