"""
Configuration utilities for AutoQA

Handles API key retrieval from environment variables or Streamlit secrets.
"""

import os
from typing import Optional


def get_api_key(key_name: str) -> Optional[str]:
    """Get API key from environment or Streamlit secrets.

    Args:
        key_name: Name of the API key (e.g., 'GLADIA_API_KEY').

    Returns:
        API key value or None if not found.
    """
    # First try environment variable
    value = os.getenv(key_name)
    if value:
        return value

    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass

    return None


def has_api_keys() -> bool:
    """Check if both required API keys are available.

    Returns:
        True if both GLADIA_API_KEY and GEMINI_API_KEY are set.
    """
    gladia = get_api_key("GLADIA_API_KEY")
    gemini = get_api_key("GEMINI_API_KEY")
    return bool(gladia and gemini)
