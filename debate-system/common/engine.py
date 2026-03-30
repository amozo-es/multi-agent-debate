"""Shared debate engine for running multi-agent sessions."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from common.utils import slugify, add_delay_between_requests
from configs.settings import Config


def run_debate_session(
    topic: str,
    moderator: Any,
    dlb: Any,
    pnm: Any,
    outpath: Path,
    decision_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run a full debate session between a moderator and two agents.

    Args:
        topic: The debate topic
        moderator: Moderator instance
        dlb: DLB Agent instance
        pnm: PNM Agent instance
        outpath: Path to save the full transcript
        decision_dir: Optional directory to save the final decision JSON

    Returns:
        Dictionary with debate results
    }
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with open(outpath, "w", encoding="utf-8") as f:
        # Start debate
        print("\n" + "="*50)
        print("DEBATE STARTED")
        print("="*50 + "\n", file=f)

        # Moderator opening
        print("Round: Opening Statement")
        mod_response = moderator.initiate_debate()
        print(f"\nModerator: {mod_response}\n", file=f)
        add_delay_between_requests(Config.SLEEP_BASE, Config.SLEEP_JITTER)

        dlb_response = ""
        pnm_response = ""

        # Debate rounds
        while True:
            print(f"\n{'='*50}")
            print(f"ROUND {moderator.round}")
            print(f"{'='*50}\n", file=f)

            # From round 2, moderator analyzes previous responses
            if moderator.round > 1:
                mod_response = moderator.analyze_and_respond(dlb_response, pnm_response)
                print(f"\nModerator: {mod_response}\n", file=f)
                add_delay_between_requests(Config.SLEEP_BASE, Config.SLEEP_JITTER)

            # Agents respond
            print("\nDLB Response:")
            dlb_response = dlb.generate(mod_response)
            print(f"\nDLB: {dlb_response}\n", file=f)
            add_delay_between_requests(Config.SLEEP_BASE, Config.SLEEP_JITTER)

            print("\nPNM Response:")
            pnm_response = pnm.generate(mod_response)
            print(f"\nPNM: {pnm_response}\n", file=f)
            add_delay_between_requests(Config.SLEEP_BASE, Config.SLEEP_JITTER)

            # Check termination
            if moderator.check_termination():
                final_text = ""
                if moderator.round > Config.MAX_ROUNDS:
                    print(f"\n{'='*50}")
                    print("FINAL ROUND - FINAL DECISION")
                    print(f"{'='*50}\n", file=f)

                    final_text = moderator.final_decision()
                    print(f"\nFinal Decision: {final_text}\n", file=f)

                    # Determine where to save the decision
                    save_dir = decision_dir if decision_dir else outpath.parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    decision_path = save_dir / f"{slugify(topic)}_decision.json"

                    decision_data = {
                        "topic": topic,
                        "final_decision": final_text
                    }

                    with open(decision_path, "w", encoding="utf-8") as jf:
                        json.dump(decision_data, jf, ensure_ascii=False, indent=2)

                    print(f"\nDecision saved to: {decision_path}")

                summary = f"Debate ended. Total rounds: {moderator.round - 1}, total messages: {moderator.total_messages}"
                print(f"\n{summary}")
                print(f"\n{summary}", file=f)
                
                return {
                    "topic": topic,
                    "rounds": moderator.round - 1,
                    "total_messages": moderator.total_messages,
                    "final_decision": final_text
                }

    return {}
