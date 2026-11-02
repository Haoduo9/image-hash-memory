"""
plugin.py - Hermes Plugin integration for Image Hash Memory & Auto-Recall.

Provides perceptual image hash memory and auto-recall capabilities for Hermes agent turns:
1. register(ctx): Registers 'pre_llm_call' and 'post_llm_call' hooks on Hermes plugin context.
2. extract_images(messages/content): Extracts and deduplicates local image file paths from
   multimodal content structures, string tags ([Image attached at: ...], [Image: ...]),
   and existing disk paths ending in image extensions.
3. pre_llm_call: Recalls visually similar images from storage (hamming distance <= 8),
   formats rich memory injection context, updates hits, or saves new images.
4. post_llm_call: Automatically extracts a short summary of assistant's visual analysis
   and backfills empty summaries in storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import yaml

try:
    from storage import (
        DEFAULT_DB_PATH,
        ImageMemoryStore,
        VisualMemoryStore,
    )
except ImportError:
    from .storage import (
        DEFAULT_DB_PATH,
        ImageMemoryStore,
        VisualMemoryStore,
    )

try:
    from hasher import (
        compute_image_fingerprint,
        hamming_distance,
        similarity_score,
    )
except ImportError:
    from .hasher import (
        compute_image_fingerprint,
        hamming_distance,
        similarity_score,
    )

logger = logging.getLogger("hermes.plugin.image_hash_memory")

__all__ = [
    "register",
    "extract_images",
    "pre_llm_call",
    "post_llm_call",
    "get_store",
    "set_store",
    "load_config",
    "IMAGE_EXTENSIONS",
]

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".ico",
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "hamming_threshold": 25,
    "enable_phash": True,
    "phash_threshold": 12,
    "enable_recall": True,
    "auto_append_summary": True,
    "max_summary_length": 200,
    "db_path": DEFAULT_DB_PATH,
}


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml with fallback to defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.is_file():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    cfg.update(loaded)
        except Exception as exc:
            logger.warning("Failed to parse config.yaml, using defaults: %s", exc)
    if "db_path" in cfg and isinstance(cfg["db_path"], str):
        cfg["db_path"] = os.path.expanduser(cfg["db_path"])
    return cfg


_GLOBAL_STORE: Optional[ImageMemoryStore] = None


def get_store(db_path: Optional[Union[str, os.PathLike]] = None) -> ImageMemoryStore:
    """
    Get or create an ImageMemoryStore instance.
    Uses db_path if specified; otherwise defaults to config db_path or DEFAULT_DB_PATH.
    """
    global _GLOBAL_STORE
    if db_path is not None:
        return ImageMemoryStore(db_path=db_path)
    if _GLOBAL_STORE is None:
        cfg = load_config()
        target_path = cfg.get("db_path") or DEFAULT_DB_PATH
        _GLOBAL_STORE = ImageMemoryStore(db_path=target_path)
    return _GLOBAL_STORE


def set_store(store: Optional[ImageMemoryStore]) -> None:
    """
    Set or reset the global visual memory store instance (useful for test isolation).
    """
    global _GLOBAL_STORE
    _GLOBAL_STORE = store


def _clean_path(path_str: str) -> str:
    """
    Strip surrounding quotes, angle brackets, and file:// protocol prefixes from a path string.
    """
    if not isinstance(path_str, str):
        return ""
    cleaned = path_str.strip().strip("\"'`<>[](){},;:")
    if cleaned.startswith("file://"):
        cleaned = cleaned[7:]
        if cleaned.startswith("localhost/"):
            cleaned = cleaned[9:]
    return cleaned


def extract_images(content_or_messages: Any) -> List[str]:
    """
    Extract and deduplicate local image file paths from:
    1. Multimodal list of parts:
       - {"type": "image_url", "image_url": {"url": "..."}} or {"type": "image_url", "image_url": "..."}
       - {"image": "..."}
       - {"path": "..."}
    2. String patterns:
       - [Image attached at: ...]
       - [Image: ...]
       - Explicit file paths ending with common image extensions that actually exist on disk (os.path.exists()).
    3. Preserves first-seen order and deduplicates paths.
    """
    extracted: List[str] = []
    seen: Set[str] = set()

    def _register_path(raw_candidate: Any, require_disk_check: bool = False) -> None:
        if not raw_candidate or not isinstance(raw_candidate, str):
            return
        cleaned = _clean_path(raw_candidate)
        if not cleaned:
            return
        # Skip web URLs or base64 inline data URIs
        if cleaned.startswith(("http://", "https://", "data:", "ftp://")):
            return

        expanded = os.path.expanduser(cleaned)
        exists_on_disk = os.path.exists(expanded) and not os.path.isdir(expanded)

        if require_disk_check and not exists_on_disk:
            return

        # Deduplication key based on normalized canonical path if it exists on disk
        if exists_on_disk:
            dedup_key = os.path.realpath(expanded)
            canonical_path = str(Path(expanded).resolve())
        else:
            dedup_key = cleaned
            canonical_path = cleaned

        if dedup_key not in seen:
            seen.add(dedup_key)
            extracted.append(canonical_path if exists_on_disk else cleaned)

    def _extract_from_text(text: str) -> None:
        if not isinstance(text, str) or not text:
            return

        # 1. Pattern: [Image attached at: ...]
        for match in re.finditer(r"\[Image attached at:\s*([^\]]+)\]", text, re.IGNORECASE):
            _register_path(match.group(1), require_disk_check=False)

        # 2. Pattern: [Image: ...]
        for match in re.finditer(r"\[Image:\s*([^\]]+)\]", text, re.IGNORECASE):
            _register_path(match.group(1), require_disk_check=False)

        # 3. Quoted strings ending in image extensions that exist on disk
        for match in re.finditer(
            r"[\"'`]([^\"'`\n\r\t]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|tif|ico))[\"'`]",
            text,
            re.IGNORECASE,
        ):
            _register_path(match.group(1), require_disk_check=True)

        # 4. Explicit path tokens ending in image extensions that exist on disk
        for token in re.split(r"[\s<>\[\]\(\)\{\}\"'`,;]+", text):
            token_clean = _clean_path(token.strip(".,;:!?"))
            if token_clean.lower().endswith(IMAGE_EXTENSIONS):
                _register_path(token_clean, require_disk_check=True)

    def _walk(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            _extract_from_text(obj)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif isinstance(obj, dict):
            # Check multimodal parts
            part_type = obj.get("type")
            if part_type == "image_url":
                iu = obj.get("image_url")
                if isinstance(iu, dict):
                    _register_path(iu.get("url"), require_disk_check=False)
                elif isinstance(iu, str):
                    _register_path(iu, require_disk_check=False)
            elif part_type in ("image", "input_image"):
                if "image" in obj:
                    val = obj["image"]
                    _register_path(val if isinstance(val, str) else (val.get("url") or val.get("path")), require_disk_check=False)
                elif "path" in obj:
                    _register_path(obj.get("path"), require_disk_check=False)
                elif "url" in obj:
                    _register_path(obj.get("url"), require_disk_check=False)

            if "image" in obj:
                img_val = obj["image"]
                if isinstance(img_val, str):
                    _register_path(img_val, require_disk_check=False)
                elif isinstance(img_val, dict):
                    _register_path(img_val.get("url") or img_val.get("path"), require_disk_check=False)

            if "path" in obj and isinstance(obj["path"], str):
                p_val = obj["path"]
                if p_val.lower().endswith(IMAGE_EXTENSIONS):
                    _register_path(p_val, require_disk_check=False)
                else:
                    _register_path(p_val, require_disk_check=True)

            if "content" in obj:
                _walk(obj["content"])
            if "text" in obj and isinstance(obj["text"], str):
                _extract_from_text(obj["text"])

    _walk(content_or_messages)
    return extracted


def _extract_summary(response: Any, max_chars: int = 200) -> str:
    """
    Extract a short summary (first 100-200 chars or first paragraph) from assistant response.
    """
    if not response:
        return ""
    if isinstance(response, str):
        text = response.strip()
    elif isinstance(response, dict):
        text = str(response.get("content", "")).strip()
    elif hasattr(response, "content"):
        text = str(getattr(response, "content", "")).strip()
    else:
        text = str(response).strip()

    if not text:
        return ""

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text[:max_chars].rstrip()

    # If first paragraph is already reasonably informative (>= 40 chars), use it directly
    first_p = paragraphs[0].replace("\n", " ").strip()
    if len(first_p) >= 40 or len(paragraphs) == 1:
        return first_p[:max_chars].rstrip()

    # Otherwise join first two short paragraphs
    joined: List[str] = [first_p]
    second_p = paragraphs[1].replace("\n", " ").strip()
    if second_p:
        joined.append(second_p)

    summary = "；".join(joined)
    if len(summary) > max_chars:
        return summary[:max_chars].rstrip()
    return summary


def register(ctx: Any) -> None:
    """
    Register image-hash-memory hooks into Hermes PluginContext:
    - pre_llm_call: Visual memory auto-recall and context injection.
    - post_llm_call: Visual analysis summary backfill.
    """
    try:
        if ctx is not None and hasattr(ctx, "register_hook"):
            ctx.register_hook("pre_llm_call", pre_llm_call)
            ctx.register_hook("post_llm_call", post_llm_call)
            logger.info("Registered image-hash-memory hooks: pre_llm_call, post_llm_call")
    except Exception as exc:
        logger.warning("Failed to register image-hash-memory hooks: %s", exc)


def pre_llm_call(
    user_message: Any = None,
    conversation_history: Optional[List[Any]] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    platform: Optional[str] = None,
    store: Optional[ImageMemoryStore] = None,
    **kwargs: Any,
) -> Optional[Dict[str, str]]:
    """
    Hermes pre_llm_call hook:
    - Extracts images from user_message / conversation_history.
    - If no images found, returns None.
    - For each image, searches visual memory via storage.find_similar().
    - If matches found:
        Formats rich memory injection context:
        [视觉记忆库命中: 该图片与历史图片高度相似（汉明距离 {d}，首次出现于 {first_seen}，累计发送 {count} 次）。历史上下文/AI记录: {ai_summary}]
        And records/updates hit in storage.
    - If no match found:
        Saves as new record via storage.record_image().
    - Returns {"context": injection_text} if matches found, otherwise None.
    """
    try:
        # Strictly extract images ONLY from the CURRENT turn's user_message!
        # Never fall back to conversation history when the user sent pure text.
        images = extract_images(user_message) if user_message is not None else []
        if not images and user_message is None and conversation_history:
            last_msg = conversation_history[-1]
            if isinstance(last_msg, dict) and last_msg.get("role") == "user":
                images = extract_images(last_msg)

        if not images:
            return None

        cfg = load_config()
        if not cfg.get("enable_recall", True):
            return None

        h_thresh = int(cfg.get("hamming_threshold", 25))
        enable_p = bool(cfg.get("enable_phash", True))
        p_thresh = int(cfg.get("phash_threshold", 12))

        mem_store = store if store is not None else get_store()
        matches_found = False
        injections: List[str] = []

        for img_path in images:
            try:
                # Find matching candidates using hamming_threshold
                matches = mem_store.find_similar(img_path, threshold=h_thresh)
                # If enable_phash is set and no dhash match found, check pHash distance
                if not matches and enable_p:
                    candidates = mem_store.find_similar(img_path, threshold=50)
                    for c in candidates:
                        if c.get("phash_distance") is not None and c["phash_distance"] <= p_thresh:
                            matches = [c]
                            break
            except Exception as match_err:
                logger.warning("Error searching similar images for %s: %s", img_path, match_err)
                matches = []

            if matches:
                matches_found = True
                best_match = matches[0]
                d = best_match.get("hamming_distance", 0)
                first_seen = (
                    best_match.get("first_seen_at")
                    or best_match.get("first_seen")
                    or "未知时间"
                )
                ai_summary = (
                    best_match.get("ai_summary")
                    or best_match.get("summary")
                    or "暂无记录"
                )

                # Within threshold <= 25: DO NOT insert new record, update existing record's hits and last_seen
                now_iso = datetime.now(timezone.utc).isoformat()
                try:
                    with mem_store._lock:
                        mem_store.conn.execute(
                            "UPDATE images SET hit_count = hit_count + 1, last_seen_at = ? WHERE id = ?",
                            (now_iso, best_match["id"]),
                        )
                        if getattr(mem_store, "auto_commit", True):
                            mem_store.conn.commit()
                except Exception as hit_err:
                    logger.warning("Error incrementing hit count on match: %s", hit_err)

                prev_count = best_match.get("seen_count") or best_match.get("hit_count") or 1
                count = prev_count + 1

                injection_line = (
                    f"[视觉记忆库命中: 该图片与历史图片高度相似（汉明距离 {d}，"
                    f"首次出现于 {first_seen}，累计发送 {count} 次）。"
                    f"历史上下文/AI记录: {ai_summary}]"
                )
                injections.append(injection_line)
            else:
                # No match found within threshold: Save as new record via storage.record_image()
                try:
                    mem_store.record_image(
                        img_path,
                        threshold=h_thresh,
                        session_id=session_id,
                        platform=platform,
                    )
                except Exception as rec_err:
                    logger.warning("Error saving new image record for %s: %s", img_path, rec_err)

        if matches_found and injections:
            return {"context": "\n\n".join(injections)}

        return None
    except Exception as exc:
        logger.warning("pre_llm_call failed open: %s", exc)
        return None


def post_llm_call(
    user_message: Any = None,
    assistant_response: Any = None,
    conversation_history: Optional[List[Any]] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    platform: Optional[str] = None,
    store: Optional[ImageMemoryStore] = None,
    **kwargs: Any,
) -> None:
    """
    Hermes post_llm_call hook:
    - Extracts images from user_message (with fallback to conversation_history).
    - If images exist and assistant_response is non-empty:
        Extracts a short summary (first 100-200 chars or first paragraph)
        and updates empty summary in storage for those images.
    """
    try:
        images = extract_images(user_message) if user_message is not None else []
        if not images and user_message is None and conversation_history:
            last_msg = conversation_history[-1]
            if isinstance(last_msg, dict) and last_msg.get("role") == "user":
                images = extract_images(last_msg)

        if not images or not assistant_response:
            return

        cfg = load_config()
        max_len = int(cfg.get("max_summary_length", 200))
        auto_append = bool(cfg.get("auto_append_summary", True))
        h_thresh = int(cfg.get("hamming_threshold", 25))

        summary = _extract_summary(assistant_response, max_chars=max_len)
        if not summary:
            return

        mem_store = store if store is not None else get_store()

        for img_path in images:
            try:
                # Find matching original image within threshold
                matches = mem_store.find_similar(img_path, threshold=h_thresh)
                if matches:
                    rec = matches[0]
                    # Update or append new summary onto existing descriptions
                    mem_store.update_summary(rec["id"], summary=summary, append=auto_append)
                else:
                    # If not recorded yet, record image with summary
                    mem_store.record_image(
                        img_path,
                        summary=summary,
                        threshold=h_thresh,
                        session_id=session_id,
                        platform=platform,
                    )
            except Exception as update_err:
                logger.warning("Failed to update summary for image %s: %s", img_path, update_err)
    except Exception as exc:
        logger.warning("post_llm_call failed open: %s", exc)
