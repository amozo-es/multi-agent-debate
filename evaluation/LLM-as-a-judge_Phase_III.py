#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 
# LLM-as-a-Judge (DeepSeek R1) reading multiple JSONs from three folders (A, B, C)
# ---------------------------------------------------------------------------------
# 
# Reads all *.json from three paths:
#   --homog_dir : JSONs from the multi-agent system ("system A")
#   --indiv_dir : JSONs from the individual LLM ("system B")
#   --heter_dir   : JSONs from the third system ("system C")
# 
# Expected item format (the file can be an object or a list of objects):
# {
#   "topic": "...",
#   "final_decision": "..."
# }
# 
# Pairs by 'topic' (exact match after strip; optional normalization with
# --normalize_topics). Sends three anonymous answers (A/B/C) to the judge. The judge
# does not see their origin.
# 
# Output: one JSON per evaluation in --outdir, the name uses a slug of the topic.
# Also writes an index.json with all results.
# 
# Example:
# python LLM-as-a-judge.py \
#   --homog_dir ./inputs/homog_decisions/ \
#   --indiv_dir ./inputs/indiv_decisions/ \
#   --heter_dir ./inputs/heter_decisions/ \
#   --outdir ./judgements
# 
# In local:
# python LLM-as-a-judge.py ^
# --homog_dir ./sistema-homogeneo/decisiones_JSON ^
# --indiv_dir ./llm-individual/decisiones_JSON ^
# --heter_dir ./sistema-heterogeneo/decisiones_JSON ^
# --outdir ./judgements

#!/usr/bin/env python3
import argparse
import json
import random
import re
import time
import traceback
import sys
from pathlib import Path
from json_repair import repair_json
from collections.abc import Iterable
from dataclasses import dataclass
from ollama_judge_client import OllamaJudgeClient

sys.stdout.reconfigure(encoding='utf-8')

# --- SATCOM Weights Configuration ---
WEIGHTS = {
    "executive_readiness": 0.35,  
    "synthesis_cohesion": 0.30,   
    "technical_specificity": 0.20,
    "constraint_alignment": 0.15  
}
HALLUCINATION_PENALTY = -0.25

def calculate_weighted_score(scores):
    """Calculates the final score based on SATCOM weights."""
    total = sum(scores[crit] * WEIGHTS[crit] for crit in WEIGHTS)
    # total += len(scores['hallucinations']) * HALLUCINATION_PENALTY
    return round(total, 2)

@dataclass
class Entry:
    topic: str
    answer: str
    src: Path


# -------- JSON loading --------

def load_json_file(path: Path) -> list[dict]:
    """
    Loads a JSON file that can be either:
      - a dict: returned as [dict]
      - a list[dict]: returned as-is
    """
    data = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON root type in {path}: {type(data)}")


def iter_entries_from_dir(dirpath: Path) -> Iterable[Entry]:
    """
    Yields Entry(topic, answer, src) for each valid item found in *.json files.
    Skips files that cannot be parsed or items lacking required fields.
    """
    for p in sorted(dirpath.glob("*.json")):
        try:
            items = load_json_file(p)
        except Exception as e:
            print(f"[warn] Could not read {p}: {e}")
            continue
        for it in items:
            topic = (it.get("topic") or "").strip()
            ans = (it.get("final_decision") or "").strip()
            if topic and ans:
                yield Entry(topic=topic, answer=ans, src=p)


# -------- Topic normalization (optional) --------

def normalize_topic(s: str) -> str:
    # lower + collapse inner spaces
    return re.sub(r"\s+", " ", s.strip().lower())


# -------- Build topic -> Entry map --------

def build_topiheter_map(entries: Iterable[Entry], normalize: bool) -> dict[str, Entry]:
    """
    Builds a dict topic_key -> Entry. If duplicates appear, keeps the first one.
    """
    m: dict[str, Entry] = {}
    for e in entries:
        key = normalize_topic(e.topic) if normalize else e.topic.strip()
        if key in m:
            print(f"[warn] Duplicate topic ignored: '{e.topic}' in {e.src}")
            continue
        m[key] = e
    return m


# -------- Utils --------

def slugify(text: str) -> str:
    """
    Safe slug from topic for filenames (ASCII-ish without requiring extra deps).
    """
    s = text.strip().lower()
    s = re.sub(r"\W+", "-", s).strip("-")
    return s[:120] or "topic"

def validate_judge_response(data, weights_dict):
    """
    Validates judge JSON using the keys from the WEIGHTS dictionary as the source of truth.
    """
    required_roots = {"reasoning", "evaluations"}
    required_answers = {"Answer_1", "Answer_2", "Answer_3"}
    # DYNAMIC: Use the keys from your WEIGHTS dict as the required attributes
    required_attributes = set(weights_dict.keys())

    # 1. Check Root Structure
    if not isinstance(data, dict) or not required_roots.issubset(data.keys()):
        return False, f"Missing root keys. Expected: {required_roots}"

    # 2. Check Answers and Attributes
    evals = data.get("evaluations", {})
    if not required_answers.issubset(evals.keys()):
        return False, f"Missing required answers. Expected: {required_answers}"

    for ans_key in required_answers:
        ans_data = evals[ans_key]
        
        # Check if answer has all keys defined in WEIGHTS
        if not required_attributes.issubset(ans_data.keys()):
            missing = required_attributes - ans_data.keys()
            return False, f"{ans_key} is missing criteria defined in WEIGHTS: {missing}"
            
        # Check score validity (0-10)
        for attr in required_attributes:
            val = ans_data.get(attr)
            if not isinstance(val, (int, float)) or not (0 <= val <= 10):
                return False, f"Invalid score in {ans_key} -> {attr}: {val}"

    return True, "Valid"

# -------- Main --------

def main():
    ap = argparse.ArgumentParser(
        description="DeepSeek R1 evaluation from three JSON folders (one file per topic)."
    )
    ap.add_argument("--homog_dir", required=True, help="Folder with system Homogeneity JSONs")
    ap.add_argument("--indiv_dir", required=True, help="Folder with individual LLM JSONs")
    ap.add_argument("--heter_dir",   required=True, help="Folder with third system Heterogeneity JSONs")
    ap.add_argument("--outdir", required=True, help="Output folder for per-topic JSONs")
    ap.add_argument("--model", default="gpt-oss:120b", help="Ollama model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--normalize_topics",
        action="store_true",
        help="Normalize topics (lower + collapse spaces) before matching",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    homog_dir = Path(args.homog_dir)
    indiv_dir = Path(args.indiv_dir)
    heter_dir   = Path(args.heter_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    homog_map = build_topiheter_map(iter_entries_from_dir(homog_dir), args.normalize_topics)
    indiv_map = build_topiheter_map(iter_entries_from_dir(indiv_dir), args.normalize_topics)
    heter_map   = build_topiheter_map(iter_entries_from_dir(heter_dir),   args.normalize_topics)
    
    # Common topics across the three systems
    common_topics = [t for t in homog_map.keys() if t in indiv_map and t in heter_map]
    if not common_topics:
        raise SystemExit("No matching topics among the three folders.")

    client = OllamaJudgeClient(args.model)
    index_results = []
    total_start_time = time.time()

    for i, topic in enumerate(common_topics, start=1):
        
        # --- SKIP BLOCK (CONTINUE) ---
        # Define the filename at the beginning to check it
        fname = f"{i:03d}_{slugify(topic)}.json"
        file_path = outdir / fname

        if file_path.exists():
            print(f"[{i}/{len(common_topics)}] SKIPPING: {topic} (File {fname} already exists)")
            continue
        # ----------------------------------
        
        topic_start = time.time() # Start of the current topic
        print(f"\n" + "="*60)
        print(f"[{i}/{len(common_topics)}] JUDGING: {topic}")
        print("="*60)
        
        # 1. Prepare 3 sources
        sources = [
            {"id": "A", "name": "Multiagent Homogeneous", "ans": homog_map[topic].answer},
            {"id": "B", "name": "Individual LLM", "ans": indiv_map[topic].answer},
            {"id": "C", "name": "Multiagent Heterogeneous", "ans": heter_map[topic].answer}
        ]
        
        # 2. SHUFFLE (Shuffle) to avoid position bias
        shuffled_indices = [0, 1, 2]
        random.shuffle(shuffled_indices)
        shuffled_sources = [sources[idx] for idx in shuffled_indices]
        
        # Mapping to know which letter (A, B, C) from the Judge corresponds to each real system
        # The judge will see "Answer 1", "Answer 2", "Answer 3"
        judge_map = {f"Answer_{j+1}": shuffled_sources[j] for j in range(3)}

        prompt_answers = ""
        for idx, item in enumerate(shuffled_sources, start=1):
            prompt_answers += f"Answer {idx}:\n{item['ans']}\n\n"

        # 3. Prompt to the Judge (only raw scores requested)
        
        JUDGE_SYSTEM = (
            "You are a decisive Chief Technology Officer (CTO) evaluating high-priority engineering briefs. "
            "You have zero tolerance for academic fluff or bureaucratic noise. "
            "Your evaluation philosophy is based on **'Decision Velocity'**: "
            "1. **Format:** You demand an immediate, visually distinct verdict at the very top (BLUF - Bottom Line Up Front). "
            "2. **Voice:** You accept only a unified, authoritative institutional voice. You reject reports that read like a transcript of a debate between internal sub-components."
        )
        
        user_prompt = (
            "You will receive a Topic and three anonymous answers (1, 2, 3).\n"
            "Evaluate the three responses (1, 2, 3) using the following criteria. For each criterion assign a natural number between 0 and 10."
            
            "CRITERIA & SCORING RULES:\n\n"

            "1. **executive_readiness (35%)** [The Format Filter]:\n"
            "   - **Goal:** Does the answer look like a Decision or an Essay?\n"
            "   - **Score 9-10:** The answer starts **immediately** with a clear, bold header (e.g., 'Final Decision' or similar) followed strictly by the solution.\n"
            "   - **Score 0-3:** The answer starts with academic preamble ('To determine...', 'In this context...'), definitions, or equations. This is a disqualification for a CTO role.\n\n"

            "2. **synthesis_cohesion (30%)** [The Voice Filter]:\n"
            "   - **Goal:** Is the answer a clean product or a 'dirty' process description?\n"
            "   - **Score 9-10 (Unified Voice):** The text presents the logic as a single, confident entity (e.g., 'The optimal approach balances X and Y...').\n"
            "   - **Score 4-5 (Procedural Leakage):** The text explicitly describes the internal discussion, mentioning 'consensus', 'agreement', 'debate', or naming internal actors/agents. **Penalize heavily**: The CTO wants the result, not the meeting minutes.\n\n"

            "3. **technical_specificity (20%)**:\n"
            "   - Does the answer propose a concrete engineering action (e.g., 'Use Digital Beamforming') rather than a vague concept?\n"
            "   - Rewards explicit mentions of trade-offs (e.g., 'Gain vs. Bandwidth').\n\n"

            "4. **constraint_alignment (15%)**:\n"
            "   - Checks if the answer explicitly references specific constraints mentioned in the prompt (e.g., 'saturated transponder', 'limited power budget').\n\n"
            f"Topic: {topic}\n\n{prompt_answers}"
            "INSTRUCTIONS:\n"
            "- **Identify the Winner:** The answer that combines the **Header** (Criteria 1) with a **Clean Voice** (Criteria 2).\n"
            "- **Identify the Losers:** Answers that lack the header (too academic) OR answers that clutter the text with 'Agent X said this' (too bureaucratic).\n"
            "- **Return ONLY a compact JSON with this EXACT schema**:\n"
            "{\n"
            "  \"reasoning\": \"Brief explanation of why the top‑ranked answer provides more strategic value than the others.\"\n"
            "  \"evaluations\": {\n"
            "    \"Answer_1\": {\n"
            "      \"executive_readiness\": <0‑10>,\n"
            "      \"synthesis_cohesion\": <0‑10>,\n"
            "      \"technical_specificity\": <0‑10>,\n"
            "      \"constraint_alignment\": <0‑10>,\n"
            "      \"hallucinations\": []\n"
            "    },\n"
            "    \"Answer_2\": { ... },\n"
            "    \"Answer_3\": { ... },\n"
            "  },\n"
            "}\n\n"
            
        )
        
        # 4. Call Local Judge
        try:
            raw_response = client.chat([
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt}
            ])
            
            res = raw_response
            print(f"\n" + "-"*30)
            print(f"📊 INFERENCE METRICS (Ollama):")
            print(f"   Prompt (Prefill): {res['prompt_tokens']} tokens in {res['prompt_sec']}s")
            print(f"   Generation:       {res['eval_tokens']} tokens in {res['eval_sec']}s")
            print(f"   Execution Speed:  {res['tps']} tokens/s")
            print("-"*30)
            
            judge = raw_response['content'].strip()
            print(f"This is the judge output (RAW):{judge}")
            # repair_json looks for the JSON within the text, fixes it if broken
            cleaned = repair_json(judge)
            
            try:
                # ---- Try parsing ----
                data = json.loads(cleaned)
                # Verify the JSON has what we need
                # Pass WEIGHTS constant directly to the validator
                is_valid, message = validate_judge_response(data, WEIGHTS)
                
                if not is_valid:
                    raise ValueError(f"Error: {message}")
                
                # 5. PYTHON CALCULATION (Avoids LLM errors)
                final_scores_per_system = {}
                for judge_key, scores in data["evaluations"].items():
                    system_info = judge_map[judge_key]
                    weighted_total = calculate_weighted_score(scores)
                    
                    final_scores_per_system[system_info["id"]] = {
                        "system": system_info["name"],
                        "weighted_total": weighted_total,
                        "raw_scores": scores
                    }
                
            except Exception as e:
                print(f"[warn] ⚠️ Error parsing judge JSON for topic '{topic}': {e}")
                # This will print the error itself
                print(traceback.print_exc(limit=1)) 
                print("Problematic content:")
                print(judge) 
                print("Cleaned problematic content:")
                print(cleaned)
                print("[warn] ⚠️ Skipping this topic and continuing with the next one.\n")
                continue  # <-- skip to next topic without stopping script
                
        except Exception as e:
            print(f"\n❌ Critical error on topic {topic}: {e}")
            # This will print the full traceback legibly
            print(traceback.format_exc()) 
            continue

        # Determine real winner in Python
        scores_only = {k: v["weighted_total"] for k, v in final_scores_per_system.items()}
        max_score = max(scores_only.values())
        winners = [k for k, v in scores_only.items() if v == max_score]
        winner_id = winners[0] if len(winners) == 1 else "tie"

        # 6. Save results
        result_obj = {
            "topic": topic,
            "sources": sources,
            "winner": winner_id,
            "final_results": final_scores_per_system,
            "reasoning": data.get("reasoning", ""),
            "shuffle_order": [s["id"] for s in shuffled_sources] # For traceability
        }
        
        index_results.append(result_obj)
        
        print("Writing response to JSON...")
        # Write one JSON per evaluation
        (file_path).write_text(
            json.dumps(result_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        index_results.append(result_obj)
        print("Response saved successfully.")
        # --- FINAL TOPIC SUMMARY ---
        topic_end = time.time()
        topic_total_duration = topic_end - topic_start
        
        print(f"-> Winner: {result_obj['winner']}")
        print(f"-> Total topic time: {topic_total_duration:.2f}s")

    # Write global index
    (outdir / "index.json").write_text(
        json.dumps(index_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Per-topic JSONs in: {outdir}\nIndex: {outdir / 'index.json'}")
    # --- GLOBAL METRICS UPON COMPLETION ---
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    avg_time = total_execution_time / len(common_topics) if common_topics else 0

    print("\n" + "#"*60)
    print(f"EXECUTION FINISHED")
    print(f"Total time: {total_execution_time/60:.2f} minutes")
    print(f"Average per topic: {avg_time:.2f} seconds")
    print("#"*60)


if __name__ == "__main__":
    main()
