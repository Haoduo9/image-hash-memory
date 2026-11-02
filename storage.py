"""
storage.py - SQLite Visual Memory Storage and Hamming Perceptual Matching.

Provides persistent perceptual image memory for AI agent workflows:
1. SQLite storage with default path ~/.hermes/data/image_memory.db.
2. Table 'images' storing hashes (dhash, ahash, phash), image geometry, metadata,
   user text, AI summary, hit counts, and timestamps.
3. User-defined function 'hamming_distance' registered on SQLite connection for
   accelerated candidate retrieval directly in SQL, with Python fallback.
4. record_image: Extracts image fingerprint via compute_image_fingerprint,
   updates existing identical image (dhash distance == 0) by incrementing hit_count,
   updating last_seen_at, and updating summary/user_text if provided; otherwise inserts
   a new record.
5. find_similar: Searches for visually matching images given a PIL Image, file path,
   bytes, or 16-char hex hash, computing Hamming distances and returning matches sorted
   by lowest distance with hamming_distance, similarity_score, and exact_match fields.
6. update_summary: Updates the AI summary / caption for a recorded image.
7. get_stats: Aggregates total_images, total_hits, earliest_seen, and latest_seen.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import unittest

from PIL import Image, ImageDraw

try:
    from hasher import (
        compute_ahash,
        compute_dhash,
        compute_image_fingerprint,
        compute_phash,
        hamming_distance,
        similarity_score,
    )
except ImportError:
    from .hasher import (
        compute_ahash,
        compute_dhash,
        compute_image_fingerprint,
        compute_phash,
        hamming_distance,
        similarity_score,
    )

__all__ = [
    "ImageMemoryStore",
    "VisualMemoryStore",
    "DEFAULT_DB_PATH",
]

DEFAULT_DB_PATH = os.path.expanduser("~/.hermes/data/image_memory.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dhash TEXT NOT NULL,
    ahash TEXT,
    phash TEXT,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    file_path TEXT,
    session_id TEXT,
    platform TEXT,
    user_text TEXT,
    summary TEXT,
    hit_count INTEGER DEFAULT 1,
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_dhash ON images(dhash);
CREATE INDEX IF NOT EXISTS idx_images_session_id ON images(session_id);
"""


def _current_timestamp() -> str:
    """Return ISO-like UTC timestamp string formatted as 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _is_hex_hash_string(val: Any, length: int = 16) -> bool:
    """Check if value is a valid hex hash string of exact length."""
    if not isinstance(val, str):
        return False
    s = val.strip()
    if len(s) != length:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in s)


def _sqlite_hamming_distance(h1: Any, h2: Any) -> int:
    """
    SQLite User-Defined Function (UDF) computing 64-bit hex Hamming distance directly in SQL.
    Returns 64 if either hash is missing or invalid.
    """
    if h1 is None or h2 is None:
        return 64
    s1 = str(h1).strip()
    s2 = str(h2).strip()
    if not s1 or not s2:
        return 64
    try:
        return (int(s1, 16) ^ int(s2, 16)).bit_count()
    except Exception:
        return 64


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Convert a sqlite3.Row object to a clean Python dictionary with aliases."""
    if row is None:
        return None
    d = dict(row)
    # Expose convenient aliases for compatibility with architecture specs
    d["user_caption"] = d.get("user_text")
    d["ai_summary"] = d.get("summary")
    d["seen_count"] = d.get("hit_count")
    d["file_size_bytes"] = d.get("file_size")
    return d


class ImageMemoryStore:
    """
    Persistent SQLite storage and Hamming perceptual retrieval engine for visual memory.
    """

    def __init__(
        self,
        db_path: Union[str, os.PathLike] = DEFAULT_DB_PATH,
        auto_commit: bool = True,
        timeout: float = 10.0,
    ) -> None:
        """
        Initialize the ImageMemoryStore with an SQLite database path.
        Default path is expanded from ~/.hermes/data/image_memory.db.
        Pass ':memory:' for a volatile in-memory test database.
        """
        self.raw_path = db_path
        self.auto_commit = auto_commit
        self.timeout = timeout
        self._lock = threading.RLock()
        self._has_hamming_udf = False

        if str(db_path) == ":memory:" or str(db_path).startswith("file:"):
            self.db_path = str(db_path)
        else:
            self.db_path = os.path.abspath(os.path.expanduser(str(db_path)))
            dir_name = os.path.dirname(self.db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

        # Optimize SQLite pragmas for WAL mode on disk files
        if self.db_path != ":memory:" and not self.db_path.startswith("file:"):
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
            except Exception:
                pass

        self._register_udf()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the underlying SQLite connection."""
        if self._conn is None:
            raise RuntimeError("Database connection is closed.")
        return self._conn

    def _register_udf(self) -> None:
        """Register the hamming_distance user-defined function in SQLite."""
        if self._conn is None:
            return
        try:
            self._conn.create_function(
                "hamming_distance",
                2,
                _sqlite_hamming_distance,
                deterministic=True,
            )
            self._has_hamming_udf = True
        except TypeError:
            try:
                self._conn.create_function(
                    "hamming_distance",
                    2,
                    _sqlite_hamming_distance,
                )
                self._has_hamming_udf = True
            except Exception:
                self._has_hamming_udf = False
        except Exception:
            self._has_hamming_udf = False

    def _init_db(self) -> None:
        """Initialize the database schema and indexes."""
        with self._lock:
            self.conn.executescript(CREATE_TABLE_SQL)
            if self.auto_commit:
                self.conn.commit()

    def record_image(
        self,
        image: Union[str, os.PathLike, Image.Image, io.BytesIO, bytes, Dict[str, Any]],
        user_text: Optional[str] = None,
        summary: Optional[str] = None,
        session_id: Optional[str] = None,
        platform: Optional[str] = None,
        file_path: Optional[str] = None,
        threshold: int = 0,
        *,
        user_caption: Optional[str] = None,
        ai_summary: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Record an image in visual memory storage.
        Uses compute_image_fingerprint to extract perceptual hashes.

        - If an exact or near-identical image (dhash distance == 0) already exists,
          increments hit_count, updates last_seen_at, updates summary/user_text
          if provided, and returns the updated record dict.
        - Otherwise inserts a new record and returns the new record dict.
        - Returns None if image cannot be processed or hashes cannot be extracted.
        """
        # Harmonize parameter aliases
        if user_text is None and user_caption is not None:
            user_text = user_caption
        if summary is None and ai_summary is not None:
            summary = ai_summary

        # Determine file_path if not explicitly provided
        resolved_file_path = file_path
        if resolved_file_path is None:
            if isinstance(image, (str, os.PathLike)):
                s_path = str(image)
                if os.path.exists(s_path):
                    resolved_file_path = os.path.abspath(s_path)
            elif isinstance(image, dict) and "file_path" in image:
                resolved_file_path = image.get("file_path")

        # Extract fingerprint/hashes
        dhash: Optional[str] = None
        ahash: Optional[str] = None
        phash: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None
        file_size: Optional[int] = None

        if isinstance(image, dict):
            dhash = image.get("dhash")
            ahash = image.get("ahash")
            phash = image.get("phash")
            width = image.get("width")
            height = image.get("height")
            file_size = image.get("file_size") or image.get("file_size_bytes")
        elif isinstance(image, str) and _is_hex_hash_string(image, 16) and not os.path.exists(image):
            dhash = image.strip().lower()
        else:
            fp = compute_image_fingerprint(image)
            if fp is None:
                return None
            dhash = fp.get("dhash")
            ahash = fp.get("ahash")
            phash = fp.get("phash")
            width = fp.get("width")
            height = fp.get("height")
            file_size = fp.get("file_size_bytes")

        if not dhash:
            return None

        dhash = str(dhash).strip().lower()
        if ahash:
            ahash = str(ahash).strip().lower()
        if phash:
            phash = str(phash).strip().lower()

        if file_size is None and resolved_file_path and os.path.isfile(resolved_file_path):
            try:
                file_size = os.path.getsize(resolved_file_path)
            except Exception:
                file_size = None

        now = _current_timestamp()

        # Check for existing identical or similar image within threshold
        with self._lock:
            existing = None
            # 1. First check exact dhash match
            cur = self.conn.execute(
                "SELECT * FROM images WHERE dhash = ? ORDER BY id ASC LIMIT 1",
                (dhash,),
            )
            existing = cur.fetchone()

            # 2. If not exact match and threshold > 0, check similar records within threshold
            if existing is None and threshold > 0:
                candidates = self.find_similar(dhash, threshold=threshold, limit=1)
                if candidates:
                    match_id = candidates[0]["id"]
                    cur = self.conn.execute(
                        "SELECT * FROM images WHERE id = ?",
                        (match_id,),
                    )
                    existing = cur.fetchone()

            if existing is not None:
                # Existing record found: increment hit_count, update last_seen_at
                update_clauses = ["hit_count = hit_count + 1", "last_seen_at = ?"]
                params: List[Any] = [now]

                if summary is not None and str(summary).strip():
                    clean_s = str(summary).strip()
                    existing_s = str(existing["summary"] or "").strip()
                    if not existing_s:
                        merged_s = clean_s
                    elif clean_s in existing_s:
                        merged_s = existing_s
                    elif existing_s in clean_s:
                        merged_s = clean_s
                    else:
                        merged_s = f"{existing_s}；{clean_s}"
                    update_clauses.append("summary = ?")
                    params.append(merged_s)

                if user_text is not None and str(user_text).strip():
                    clean_u = str(user_text).strip()
                    existing_u = str(existing["user_text"] or "").strip()
                    if not existing_u:
                        merged_u = clean_u
                    elif clean_u in existing_u:
                        merged_u = existing_u
                    else:
                        merged_u = f"{existing_u}；{clean_u}"
                    update_clauses.append("user_text = ?")
                    params.append(merged_u)

                if session_id is not None:
                    update_clauses.append("session_id = ?")
                    params.append(session_id)

                if platform is not None:
                    update_clauses.append("platform = ?")
                    params.append(platform)

                if resolved_file_path is not None and not existing["file_path"]:
                    update_clauses.append("file_path = ?")
                    params.append(resolved_file_path)

                if existing["ahash"] is None and ahash is not None:
                    update_clauses.append("ahash = ?")
                    params.append(ahash)

                if existing["phash"] is None and phash is not None:
                    update_clauses.append("phash = ?")
                    params.append(phash)

                if existing["width"] is None and width is not None:
                    update_clauses.append("width = ?")
                    params.append(width)

                if existing["height"] is None and height is not None:
                    update_clauses.append("height = ?")
                    params.append(height)

                if existing["file_size"] is None and file_size is not None:
                    update_clauses.append("file_size = ?")
                    params.append(file_size)

                params.append(existing["id"])
                sql = f"UPDATE images SET {', '.join(update_clauses)} WHERE id = ?"
                self.conn.execute(sql, params)
                if self.auto_commit:
                    self.conn.commit()

                cur = self.conn.execute(
                    "SELECT * FROM images WHERE id = ?",
                    (existing["id"],),
                )
                updated_row = cur.fetchone()
                return _row_to_dict(updated_row)
            else:
                # Insert new image record
                cur = self.conn.execute(
                    """
                    INSERT INTO images (
                        dhash, ahash, phash,
                        width, height, file_size,
                        file_path, session_id, platform,
                        user_text, summary,
                        hit_count, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        dhash,
                        ahash,
                        phash,
                        width,
                        height,
                        file_size,
                        resolved_file_path,
                        session_id,
                        platform,
                        user_text,
                        summary,
                        now,
                        now,
                    ),
                )
                if self.auto_commit:
                    self.conn.commit()

                new_id = cur.lastrowid
                cur = self.conn.execute(
                    "SELECT * FROM images WHERE id = ?",
                    (new_id,),
                )
                new_row = cur.fetchone()
                return _row_to_dict(new_row)

    # Method aliases for record_image
    record = record_image
    store_image = record_image
    add_image = record_image
    save_image = record_image

    def find_similar(
        self,
        image_or_hash: Union[str, os.PathLike, Image.Image, io.BytesIO, bytes, Dict[str, Any]],
        threshold: int = 8,
        limit: Optional[int] = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search visual memory for matching or similar images.
        Can take an image path / PIL Image, raw bytes, or directly a 16-char hex hash.

        Calculates Hamming distance for dhash (and phash if available).
        Returns list of matching dicts sorted by lowest hamming_distance, with added fields:
          - hamming_distance: int bit difference
          - similarity_score: float score from 0.0 to 1.0
          - exact_match: bool True if hamming_distance == 0
          (and aliases: distance, dhash_distance, phash_distance, similarity, is_exact).
        """
        query_dhash: Optional[str] = None
        query_phash: Optional[str] = None

        if isinstance(image_or_hash, dict):
            query_dhash = image_or_hash.get("dhash")
            query_phash = image_or_hash.get("phash")
        elif isinstance(image_or_hash, str) and _is_hex_hash_string(image_or_hash, 16) and not os.path.exists(image_or_hash):
            query_dhash = image_or_hash.strip().lower()
        else:
            fp = compute_image_fingerprint(image_or_hash)
            if fp is not None:
                query_dhash = fp.get("dhash")
                query_phash = fp.get("phash")

        if not query_dhash:
            return []

        query_dhash = str(query_dhash).strip().lower()
        if query_phash:
            query_phash = str(query_phash).strip().lower()

        # Retrieve candidates using SQLite UDF or Python fallback
        with self._lock:
            used_udf = False
            rows: Sequence[sqlite3.Row] = []

            if self._has_hamming_udf:
                try:
                    if threshold is not None:
                        sql = """
                            SELECT *, hamming_distance(dhash, ?) AS _calc_dist
                            FROM images
                            WHERE hamming_distance(dhash, ?) <= ?
                            ORDER BY _calc_dist ASC
                        """
                        params = [query_dhash, query_dhash, threshold]
                        if limit is not None and not query_phash:
                            sql += " LIMIT ?"
                            params.append(limit)
                        rows = self.conn.execute(sql, params).fetchall()
                    else:
                        sql = """
                            SELECT *, hamming_distance(dhash, ?) AS _calc_dist
                            FROM images
                            ORDER BY _calc_dist ASC
                        """
                        params = [query_dhash]
                        if limit is not None and not query_phash:
                            sql += " LIMIT ?"
                            params.append(limit)
                        rows = self.conn.execute(sql, params).fetchall()
                    used_udf = True
                except Exception:
                    used_udf = False

            if not used_udf:
                rows = self.conn.execute("SELECT * FROM images").fetchall()

        # Build annotated match list
        matches: List[Dict[str, Any]] = []
        for r in rows:
            rec = _row_to_dict(r)
            if rec is None:
                continue
            rec.pop("_calc_dist", None)

            cand_dhash = rec.get("dhash")
            if not cand_dhash:
                continue

            try:
                dist_d = hamming_distance(cand_dhash, query_dhash)
            except Exception:
                continue

            if threshold is not None and dist_d > threshold:
                continue

            dist_p = None
            cand_phash = rec.get("phash")
            if query_phash and cand_phash:
                try:
                    dist_p = hamming_distance(cand_phash, query_phash)
                except Exception:
                    dist_p = None

            score = similarity_score(cand_dhash, query_dhash)
            is_exact = (dist_d == 0)

            rec["hamming_distance"] = dist_d
            rec["similarity_score"] = float(score)
            rec["exact_match"] = bool(is_exact)

            # Compatibility aliases
            rec["distance"] = dist_d
            rec["dhash_distance"] = dist_d
            if dist_p is not None:
                rec["phash_distance"] = dist_p
            rec["similarity"] = float(score)
            rec["is_exact"] = bool(is_exact)

            matches.append(rec)

        # Sort matches by lowest hamming_distance (and secondary phash distance)
        matches.sort(
            key=lambda m: (
                m["hamming_distance"],
                m.get("phash_distance") if m.get("phash_distance") is not None else 0,
                -m.get("hit_count", 0),
            )
        )

        if limit is not None:
            matches = matches[:limit]

        return matches

    # Method aliases for find_similar
    search = find_similar
    find = find_similar

    def update_summary(
        self,
        record_id: int,
        summary: Optional[str] = None,
        *,
        user_text: Optional[str] = None,
        ai_summary: Optional[str] = None,
        user_caption: Optional[str] = None,
        append: bool = True,
    ) -> bool:
        """
        Update the summary and/or user text for a specific image record.
        If append=True, stacks the new summary/user_text onto existing descriptions separated by '；'.
        Returns True if the record was updated, False if the record ID does not exist.
        """
        if summary is None and ai_summary is not None:
            summary = ai_summary
        if user_text is None and user_caption is not None:
            user_text = user_caption

        if summary is None and user_text is None:
            return False

        now = _current_timestamp()
        with self._lock:
            cur = self.conn.execute("SELECT * FROM images WHERE id = ?", (record_id,))
            existing = cur.fetchone()
            if existing is None:
                return False

            update_clauses = ["last_seen_at = ?"]
            params: List[Any] = [now]

            if summary is not None and str(summary).strip():
                clean_s = str(summary).strip()
                existing_s = str(existing["summary"] or "").strip()
                if not append or not existing_s:
                    merged_s = clean_s
                elif clean_s in existing_s:
                    merged_s = existing_s
                elif existing_s in clean_s:
                    merged_s = clean_s
                else:
                    merged_s = f"{existing_s}；{clean_s}"
                update_clauses.append("summary = ?")
                params.append(merged_s)

            if user_text is not None and str(user_text).strip():
                clean_u = str(user_text).strip()
                existing_u = str(existing["user_text"] or "").strip()
                if not append or not existing_u:
                    merged_u = clean_u
                elif clean_u in existing_u:
                    merged_u = existing_u
                else:
                    merged_u = f"{existing_u}；{clean_u}"
                update_clauses.append("user_text = ?")
                params.append(merged_u)

            params.append(record_id)
            sql = f"UPDATE images SET {', '.join(update_clauses)} WHERE id = ?"
            cur = self.conn.execute(sql, params)
            if self.auto_commit:
                self.conn.commit()
            return cur.rowcount > 0

    update_ai_summary = update_summary

    def get_stats(self) -> Dict[str, Any]:
        """
        Return visual memory store statistics:
          total_images: Total distinct image records in storage
          total_hits: Sum of all image hit counts
          earliest_seen: Timestamp of earliest first_seen_at record
          latest_seen: Timestamp of latest last_seen_at record
        """
        sql = """
            SELECT
                COUNT(*) AS total_images,
                COALESCE(SUM(hit_count), 0) AS total_hits,
                MIN(first_seen_at) AS earliest_seen,
                MAX(last_seen_at) AS latest_seen
            FROM images
        """
        with self._lock:
            row = self.conn.execute(sql).fetchone()
            if row is None:
                return {
                    "total_images": 0,
                    "total_hits": 0,
                    "earliest_seen": None,
                    "latest_seen": None,
                }
            return {
                "total_images": int(row["total_images"] or 0),
                "total_hits": int(row["total_hits"] or 0),
                "earliest_seen": row["earliest_seen"],
                "latest_seen": row["latest_seen"],
            }

    stats = get_stats

    def get_image(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single image record by ID, or None if not found."""
        with self._lock:
            cur = self.conn.execute("SELECT * FROM images WHERE id = ?", (record_id,))
            row = cur.fetchone()
            return _row_to_dict(row)

    get_by_id = get_image

    def get_by_dhash(self, dhash: str) -> Optional[Dict[str, Any]]:
        """Fetch a single image record by dhash, or None if not found."""
        clean = dhash.strip().lower()
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM images WHERE dhash = ? LIMIT 1",
                (clean,),
            )
            row = cur.fetchone()
            return _row_to_dict(row)

    def delete_image(self, record_id: int) -> bool:
        """Delete an image record by ID. Returns True if deleted, False otherwise."""
        with self._lock:
            cur = self.conn.execute("DELETE FROM images WHERE id = ?", (record_id,))
            if self.auto_commit:
                self.conn.commit()
            return cur.rowcount > 0

    delete_by_id = delete_image

    def delete_all(self) -> int:
        """Delete all image records in visual memory. Returns the number of rows deleted."""
        with self._lock:
            cur = self.conn.execute("DELETE FROM images")
            if self.auto_commit:
                self.conn.commit()
            return cur.rowcount

    clear = delete_all

    def count(self) -> int:
        """Return total number of unique images stored."""
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()
            return int(row[0]) if row else 0

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __enter__(self) -> "ImageMemoryStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Top-level class alias
VisualMemoryStore = ImageMemoryStore


# ---------------------------------------------------------------------------
# Unit tests using in-memory SQLite database (:memory:)
# ---------------------------------------------------------------------------
class TestImageMemoryStore(unittest.TestCase):
    """Thorough unit tests for SQLite visual memory storage and Hamming matching."""

    def setUp(self) -> None:
        self.store = ImageMemoryStore(":memory:")

        # Create distinct test synthetic PIL images
        self.img1 = Image.new("RGB", (120, 120), (255, 255, 255))
        d1 = ImageDraw.Draw(self.img1)
        d1.rectangle([10, 10, 60, 60], fill=(200, 20, 20))
        d1.line([0, 0, 120, 120], fill=(0, 0, 0), width=3)

        self.img2 = Image.new("RGB", (120, 120), (240, 240, 240))
        d2 = ImageDraw.Draw(self.img2)
        d2.ellipse([30, 30, 90, 90], fill=(20, 120, 220))
        d2.rectangle([80, 80, 110, 110], fill=(10, 200, 30))

        self.img3 = Image.new("RGB", (120, 120), (10, 10, 10))
        d3 = ImageDraw.Draw(self.img3)
        d3.polygon([(10, 10), (110, 20), (60, 110)], fill=(255, 255, 0))

    def tearDown(self) -> None:
        self.store.close()

    def test_init_and_empty_stats(self) -> None:
        """Verify schema creation and stats on an empty database."""
        stats = self.store.get_stats()
        self.assertEqual(stats["total_images"], 0)
        self.assertEqual(stats["total_hits"], 0)
        self.assertIsNone(stats["earliest_seen"])
        self.assertIsNone(stats["latest_seen"])
        self.assertEqual(self.store.count(), 0)

    def test_hamming_distance_sqlite_udf(self) -> None:
        """Verify SQLite user-defined function computes correct Hamming distances."""
        # Query UDF directly from SQL
        row0 = self.store.conn.execute(
            "SELECT hamming_distance(?, ?)",
            ("0000000000000000", "0000000000000000"),
        ).fetchone()
        self.assertEqual(row0[0], 0)

        row1 = self.store.conn.execute(
            "SELECT hamming_distance(?, ?)",
            ("0000000000000000", "0000000000000001"),
        ).fetchone()
        self.assertEqual(row1[0], 1)

        row64 = self.store.conn.execute(
            "SELECT hamming_distance(?, ?)",
            ("0000000000000000", "ffffffffffffffff"),
        ).fetchone()
        self.assertEqual(row64[0], 64)

    def test_record_new_image(self) -> None:
        """Verify inserting a new image record returns dict with all expected fields."""
        rec = self.store.record_image(
            self.img1,
            user_text="A red rectangle diagram",
            summary="Technical architectural box drawing",
            session_id="sess_alpha",
            platform="desktop",
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec["id"], 1)
        self.assertEqual(len(rec["dhash"]), 16)
        self.assertEqual(len(rec["ahash"]), 16)
        self.assertEqual(len(rec["phash"]), 16)
        self.assertEqual(rec["width"], 120)
        self.assertEqual(rec["height"], 120)
        self.assertEqual(rec["hit_count"], 1)
        self.assertEqual(rec["user_text"], "A red rectangle diagram")
        self.assertEqual(rec["summary"], "Technical architectural box drawing")
        self.assertEqual(rec["session_id"], "sess_alpha")
        self.assertEqual(rec["platform"], "desktop")
        self.assertIsNotNone(rec["first_seen_at"])
        self.assertIsNotNone(rec["last_seen_at"])
        self.assertEqual(rec["first_seen_at"], rec["last_seen_at"])

        # Check stats
        stats = self.store.get_stats()
        self.assertEqual(stats["total_images"], 1)
        self.assertEqual(stats["total_hits"], 1)
        self.assertEqual(stats["earliest_seen"], rec["first_seen_at"])
        self.assertEqual(stats["latest_seen"], rec["last_seen_at"])

    def test_record_duplicate_increments_hit_count(self) -> None:
        """Verify identical image increments hit_count and updates summary/user_text."""
        rec1 = self.store.record_image(
            self.img1,
            user_text="First description",
            summary="First summary",
        )
        self.assertEqual(rec1["hit_count"], 1)
        t1_first = rec1["first_seen_at"]

        # Record identical image again with updated summary (descriptions are stacked)
        rec2 = self.store.record_image(
            self.img1,
            user_text="Updated description",
            summary="Updated summary",
        )
        self.assertEqual(rec2["id"], rec1["id"])
        self.assertEqual(rec2["hit_count"], 2)
        self.assertEqual(rec2["first_seen_at"], t1_first)
        self.assertEqual(rec2["user_text"], "First description；Updated description")
        self.assertEqual(rec2["summary"], "First summary；Updated summary")

        # Record third time with no summary provided (should preserve existing stacked summary)
        rec3 = self.store.record_image(self.img1)
        self.assertEqual(rec3["hit_count"], 3)
        self.assertEqual(rec3["summary"], "First summary；Updated summary")
        self.assertEqual(rec3["user_text"], "First description；Updated description")

        # Verify stats reflect 1 image and 3 total hits
        stats = self.store.get_stats()
        self.assertEqual(stats["total_images"], 1)
        self.assertEqual(stats["total_hits"], 3)

    def test_record_with_hex_hash_directly(self) -> None:
        """Verify recording directly with a 16-character hex hash."""
        hex_hash = "abcdef0123456789"
        rec = self.store.record_image(
            hex_hash,
            user_text="Direct hex input",
            summary="Direct hex test summary",
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec["dhash"], hex_hash)
        self.assertEqual(rec["hit_count"], 1)

        # Re-record with same hash
        rec2 = self.store.record_image(hex_hash)
        self.assertEqual(rec2["id"], rec["id"])
        self.assertEqual(rec2["hit_count"], 2)

    def test_find_similar_with_hex_hash(self) -> None:
        """Verify finding similar records using 16-char hex hash."""
        h0 = "0000000000000000"
        h1 = "0000000000000001"  # dist 1
        h3 = "0000000000000007"  # dist 3
        h_far = "ffffffffffffffff"  # dist 64

        self.store.record_image(h0, summary="zero hash")
        self.store.record_image(h1, summary="one bit diff")
        self.store.record_image(h3, summary="three bits diff")
        self.store.record_image(h_far, summary="far away")

        matches = self.store.find_similar(h0, threshold=8)
        self.assertEqual(len(matches), 3)

        # Sorted by lowest hamming_distance
        self.assertEqual(matches[0]["dhash"], h0)
        self.assertEqual(matches[0]["hamming_distance"], 0)
        self.assertEqual(matches[0]["similarity_score"], 1.0)
        self.assertTrue(matches[0]["exact_match"])

        self.assertEqual(matches[1]["dhash"], h1)
        self.assertEqual(matches[1]["hamming_distance"], 1)
        self.assertAlmostEqual(matches[1]["similarity_score"], 63.0 / 64.0)
        self.assertFalse(matches[1]["exact_match"])

        self.assertEqual(matches[2]["dhash"], h3)
        self.assertEqual(matches[2]["hamming_distance"], 3)
        self.assertFalse(matches[2]["exact_match"])

        # Test threshold=1
        matches_strict = self.store.find_similar(h0, threshold=1)
        self.assertEqual(len(matches_strict), 2)
        self.assertEqual([m["dhash"] for m in matches_strict], [h0, h1])

        # Test threshold=0 (exact only)
        matches_exact = self.store.find_similar(h0, threshold=0)
        self.assertEqual(len(matches_exact), 1)
        self.assertEqual(matches_exact[0]["dhash"], h0)
        self.assertTrue(matches_exact[0]["exact_match"])

    def test_find_similar_with_pil_images(self) -> None:
        """Verify finding visually similar images with PIL Image objects."""
        rec1 = self.store.record_image(self.img1, summary="img1 red box")
        rec2 = self.store.record_image(self.img2, summary="img2 blue circle")
        rec3 = self.store.record_image(self.img3, summary="img3 dark yellow triangle")

        # Query with img1 exact
        matches = self.store.find_similar(self.img1, threshold=8)
        self.assertGreaterEqual(len(matches), 1)
        best = matches[0]
        self.assertEqual(best["id"], rec1["id"])
        self.assertEqual(best["hamming_distance"], 0)
        self.assertEqual(best["similarity_score"], 1.0)
        self.assertTrue(best["exact_match"])
        self.assertEqual(best["phash_distance"], 0)

        # Query with slightly modified img1 (slight line tweak)
        modified = self.img1.copy()
        draw = ImageDraw.Draw(modified)
        draw.line([0, 10, 10, 0], fill=(255, 0, 0), width=1)
        matches_mod = self.store.find_similar(modified, threshold=8)
        self.assertGreaterEqual(len(matches_mod), 1)
        self.assertEqual(matches_mod[0]["id"], rec1["id"])
        self.assertLessEqual(matches_mod[0]["hamming_distance"], 4)

    def test_python_fallback_matching(self) -> None:
        """Verify Python candidate filtering fallback produces identical results when UDF is disabled."""
        h0 = "0000000000000000"
        h1 = "0000000000000001"
        self.store.record_image(h0)
        self.store.record_image(h1)

        # Force Python fallback branch
        self.store._has_hamming_udf = False

        matches = self.store.find_similar(h0, threshold=8)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0]["hamming_distance"], 0)
        self.assertEqual(matches[0]["exact_match"], True)
        self.assertEqual(matches[1]["hamming_distance"], 1)
        self.assertEqual(matches[1]["exact_match"], False)

    def test_update_summary(self) -> None:
        """Verify updating summary and user_text for an existing image record."""
        rec = self.store.record_image(self.img1, summary="Initial summary")
        rec_id = rec["id"]

        success = self.store.update_summary(
            rec_id,
            summary="Refined summary by Hermes AI",
            user_text="User added text later",
            append=False,
        )
        self.assertTrue(success)

        updated = self.store.get_image(rec_id)
        self.assertIsNotNone(updated)
        self.assertEqual(updated["summary"], "Refined summary by Hermes AI")
        self.assertEqual(updated["user_text"], "User added text later")

        # Non-existent ID returns False
        self.assertFalse(self.store.update_summary(999999, "No such image"))

    def test_get_and_delete_image(self) -> None:
        """Verify getting and deleting image records."""
        rec1 = self.store.record_image(self.img1)
        rec2 = self.store.record_image(self.img2)
        self.assertEqual(self.store.count(), 2)

        # Get by id
        fetched = self.store.get_image(rec1["id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["id"], rec1["id"])

        # Get by dhash
        by_hash = self.store.get_by_dhash(rec1["dhash"])
        self.assertIsNotNone(by_hash)
        self.assertEqual(by_hash["id"], rec1["id"])

        # Delete image
        self.assertTrue(self.store.delete_image(rec1["id"]))
        self.assertIsNone(self.store.get_image(rec1["id"]))
        self.assertEqual(self.store.count(), 1)
        self.assertFalse(self.store.delete_image(rec1["id"]))

        # Delete all / clear
        self.assertEqual(self.store.clear(), 1)
        self.assertEqual(self.store.count(), 0)

    def test_stats_aggregation(self) -> None:
        """Verify total_images, total_hits, earliest_seen, and latest_seen."""
        self.store.record_image(self.img1)
        self.store.record_image(self.img1)
        self.store.record_image(self.img1)  # img1: 3 hits

        self.store.record_image(self.img2)
        self.store.record_image(self.img2)  # img2: 2 hits

        self.store.record_image(self.img3)  # img3: 1 hit

        stats = self.store.get_stats()
        self.assertEqual(stats["total_images"], 3)
        self.assertEqual(stats["total_hits"], 6)
        self.assertIsNotNone(stats["earliest_seen"])
        self.assertIsNotNone(stats["latest_seen"])
        self.assertLessEqual(stats["earliest_seen"], stats["latest_seen"])

    def test_corrupted_and_invalid_inputs(self) -> None:
        """Verify invalid or corrupted inputs fail gracefully without throwing exceptions."""
        self.assertIsNone(self.store.record_image(b"not an image file"))
        self.assertIsNone(self.store.record_image(None))  # type: ignore
        self.assertIsNone(self.store.record_image("non_existent_file.png"))

        self.assertEqual(self.store.find_similar(b"not an image"), [])
        self.assertEqual(self.store.find_similar(None), [])  # type: ignore
        self.assertEqual(self.store.find_similar("not_a_valid_hash_or_file"), [])

    def test_file_persistence_and_context_manager(self) -> None:
        """Verify file-based SQLite database persistence and context manager support."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "nested", "test_memory.db")

            with ImageMemoryStore(db_file) as store:
                store.record_image(self.img1, summary="Persistent item")
                self.assertEqual(store.count(), 1)

            # Re-open the database from the same file
            with ImageMemoryStore(db_file) as store2:
                self.assertEqual(store2.count(), 1)
                matches = store2.find_similar(self.img1)
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0]["summary"], "Persistent item")


if __name__ == "__main__":
    print("Running storage.py visual memory storage unit tests...")
    unittest.main()
