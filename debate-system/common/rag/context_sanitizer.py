"""Context sanitization utilities to remove references and papers."""

import re
from typing import List


# Patterns to identify and remove references, papers, and citations
REF_PATTERNS: List[str] = [
    r"\b(19|20)\d{2}\b",                    # años 1990..2099
    r"\bdoi:\S+",                           # DOI
    r"\barXiv:\S+",                         # arXiv
    r"https?://\S+",                        # URLs
    r"\b(et al\.|pp\.|Vol\.|No\.)\b",       # marcadores típicos
    r"\[\d+\]",                             # [1], [2]...
    r"\([A-Z][a-zA-Z]+,?\s*(19|20)\d{2}\)", # (Autor, 2019)
    r"\breferences\b.*",                    # sección "References" y lo que siga
    r"\bthis\s+(paper|work|study)\b.*?(?:\.|\n)", # "this paper …"
]


def sanitize_context(text: str) -> str:
    """
    Sanitize text by removing references, papers, and citations.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    # corta "References" y lo que siga
    text = re.split(r"\n\s*references\s*:?\s*$", text, flags=re.I)[0]

    # Remove patterns
    for pat in REF_PATTERNS:
        text = re.sub(pat, "", text)

    # quita "this paper …", "in this work …"
    text = re.sub(r"\b(this\s+paper|in\s+this\s+work|we\s+propose)\b.*?(?:\.|\n)", "", text, flags=re.I)

    # normaliza espacios
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate_context(text: str, max_length: int = 2000) -> str:
    """
    Truncate context to maximum length.

    Args:
        text: Input text to truncate
        max_length: Maximum length in characters

    Returns:
        Truncated text
    """
    return text[:max_length] if len(text) > max_length else text