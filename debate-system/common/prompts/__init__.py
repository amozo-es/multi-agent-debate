"""Prompts module for multi-agent satellite communication systems."""

from .agent_prompts import (
    get_moderator_system_prompt,
    get_dlb_system_prompt,
    get_pnm_system_prompt,
    get_rag_agent_system_prompt
)
from .judge_prompts import (
    get_judge_system_prompt,
    get_judge_user_prompt
)

__all__ = [
    'get_moderator_system_prompt',
    'get_dlb_system_prompt',
    'get_pnm_system_prompt',
    'get_rag_agent_system_prompt',
    'get_judge_system_prompt',
    'get_judge_user_prompt'
]