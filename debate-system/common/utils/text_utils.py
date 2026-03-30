"""Text processing utilities."""

import unicodedata
import string
from typing import List, Dict, Optional


def slugify(text: str, maxlen: int = 60) -> str:
    """
    Convert text to a URL-friendly slug.

    Args:
        text: Input text to slugify
        maxlen: Maximum length of the slug

    Returns:
        URL-friendly slug
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    allowed = f"-_.{string.ascii_letters}{string.digits}"
    text = "".join(c if c in allowed or c.isspace() else "-" for c in text).strip()
    text = "-".join(text.split())
    return (text[:maxlen] or "topic").strip("-_.")


def keep_recent_messages(messages: List[Dict[str, str]], max_messages: int = 12) -> List[Dict[str, str]]:
    """
    Keep only the most recent messages from the conversation history.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_messages: Maximum number of messages to keep (excluding system message)

    Returns:
        Filtered message list
    """
    if not messages or max_messages is None or max_messages <= 0:
        return messages

    head = messages[:1] if messages[0].get("role") == "system" else []
    tail = messages[1:] if head else messages

    if len(tail) <= max_messages:
        return messages  # no recorta si no hace falta

    tail = tail[-max_messages:]  # últimas N
    return head + tail