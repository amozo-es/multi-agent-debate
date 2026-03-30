"""Prompts for the LLM-as-a-judge evaluation system."""

def get_judge_system_prompt() -> str:
    """Get the judge system prompt."""
    return (
        "You are an impartial, rigorous evaluator. You must judge three anonymous answers "
        "(A, B and C) to the SAME question. Do NOT infer or mention who wrote each answer. "
        "Evaluate ONLY content quality."
    )

def get_judge_user_prompt(topic: str, answer_a: str, answer_b: str, answer_c: str) -> str:
    """Generate the judge user prompt for evaluating three answers."""
    return (
        "You will receive a Topic and three anonymous answers (A, B, C).\n"
        "Evaluate ONLY using the criteria that appear in the final JSON.\n"
        "Assign 0..10 to each criterion per answer and compute a weighted total with SATCOM-oriented weights:\n\n"
        "Weights (SATCOM context):\n"
        "- accuracy: 45%          (RF/mmWave correctness, link-budget terms, interference models, standards/constraints)\n"
        "- practical-viability: 25% (implementability under power, bandwidth, payload, synchronization, and coordination limits)\n"
        "- completeness: 15%       (covers the necessary technical aspects and trade-offs)\n"
        "- question-alignment: 15% (directly addresses the asked topic; strategy discussion influenced by the LPF counts as aligned)\n"
        f"Topic: {topic}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        f"Answer C:\n{answer_c}\n\n"
        "The winner MUST be the answer with the highest weighted total score. If two answers have equal total score, return 'tie'. Do NOT select a lower-scoring answer even if it seems 'more direct'.\n"
        "Do NOT penalize references or mentions to 'DLB' or 'PNM' if they are used in a technically consistent or contextually relevant manner.\n"
        "Return ONLY a compact JSON with this schema (no extra text):\n"
        "{\n"
        "  \"winner\": \"A|B|C|tie\",\n"
        "  \"scores\": { \"A\": <0..10>, \"B\": <0..10>, \"C\": <0..10> },\n"
        "  \"reasoning\": \"<=150 words with concrete, neutral justification\",\n"
        "  \"hallucinations\": { \"A\": [\"...\"], \"B\": [\"...\"], \"C\": [\"...\"] },\n"
        "  \"completeness\": { \"A\": <0..10>, \"B\": <0..10>, \"C\": <0..10> },\n"
        "  \"accuracy\": { \"A\": <0..10>, \"B\": <0..10>, \"C\": <0..10> },\n"
        "  \"practical-viability\": { \"A\": <0..10>, \"B\": <0..10>, \"C\": <0..10> },\n"
        "  \"question-alignment\": { \"A\": <0..10>, \"B\": <0..10>, \"C\": <0..10> }\n"
        "}\n"
    )