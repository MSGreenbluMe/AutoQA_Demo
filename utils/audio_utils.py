"""
Audio Utility Functions

This module provides helper functions for audio file validation
and processing.
"""

import os
from typing import Tuple, Optional

# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_audio_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validate uploaded audio file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if uploaded_file is None:
        return False, "Žiadny súbor nebol nahraný."

    # Check file extension
    filename = uploaded_file.name.lower()
    extension = os.path.splitext(filename)[1]

    if extension not in SUPPORTED_FORMATS:
        return False, (
            f"Nepodporovaný formát súboru: {extension}. "
            f"Podporované formáty: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check file size
    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return False, (
            f"Súbor je príliš veľký: {size_mb:.1f} MB. "
            f"Maximálna veľkosť: {MAX_FILE_SIZE_MB} MB"
        )

    return True, None


def get_file_info(uploaded_file) -> dict:
    """Get information about uploaded file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Dictionary with file information.
    """
    if uploaded_file is None:
        return {}

    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()
    size_bytes = uploaded_file.size
    size_mb = size_bytes / (1024 * 1024)

    return {
        "name": filename,
        "extension": extension,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "type": uploaded_file.type,
    }


def format_file_size(size_bytes: int) -> str:
    """Format file size to human readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted size string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.
    """
    if seconds < 0:
        return "00:00"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    if minutes >= 60:
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def get_language_name(code: str) -> str:
    """Get language name from code.

    Args:
        code: Language code (e.g., 'sk', 'cs', 'en').

    Returns:
        Language name in Slovak.
    """
    languages = {
        "sk": "Slovenčina",
        "cs": "Čeština",
        "en": "Angličtina",
        "de": "Nemčina",
        "hu": "Maďarčina",
        "pl": "Poľština",
        "auto": "Automatická detekcia",
    }
    return languages.get(code, code)
