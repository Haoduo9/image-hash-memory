"""
image-hash-memory - Perceptual Image Hash Memory & Auto-Recall for Hermes.
"""

from .plugin import (
    IMAGE_EXTENSIONS,
    extract_images,
    get_store,
    post_llm_call,
    pre_llm_call,
    register,
    set_store,
)

__all__ = [
    "register",
    "extract_images",
    "pre_llm_call",
    "post_llm_call",
    "get_store",
    "set_store",
    "IMAGE_EXTENSIONS",
]
