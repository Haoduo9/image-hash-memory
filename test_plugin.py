"""
test_plugin.py - Unit tests for Hermes image-hash-memory plugin integration.

Verifies:
1. Mock register() hook registration.
2. extract_images helper with multimodal parts, string patterns, disk existence checks,
   and deduplication.
3. pre_llm_call hook with no images, different images, identical images (recall + hit update),
   and similar image matching.
4. post_llm_call hook with summary extraction, backfilling empty summary, preserving
   existing summary, and handling empty/invalid responses.
5. Fail-open error handling.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock

from PIL import Image, ImageDraw

from plugin import (
    IMAGE_EXTENSIONS,
    extract_images,
    get_store,
    post_llm_call,
    pre_llm_call,
    register,
    set_store,
)
from storage import ImageMemoryStore


class TestMockRegister(unittest.TestCase):
    """Test Hermes plugin register hook integration."""

    def test_mock_register(self):
        """register(ctx) should register both pre_llm_call and post_llm_call hooks."""
        mock_ctx = MagicMock()
        register(mock_ctx)

        registered_calls = {call[0][0]: call[0][1] for call in mock_ctx.register_hook.call_args_list}
        self.assertIn("pre_llm_call", registered_calls)
        self.assertIn("post_llm_call", registered_calls)

        self.assertEqual(registered_calls["pre_llm_call"], pre_llm_call)
        self.assertEqual(registered_calls["post_llm_call"], post_llm_call)

    def test_register_safe_with_invalid_context(self):
        """register() should fail open and not raise if ctx is None or lacks register_hook."""
        try:
            register(None)
            register(object())
        except Exception as exc:
            self.fail(f"register() raised unexpected exception on invalid ctx: {exc}")


class TestExtractImages(unittest.TestCase):
    """Test multimodal and string pattern image path extraction and deduplication."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

        # Create real sample image files on disk
        self.img1_path = str(self.dir_path / "img1.png")
        self.img2_path = str(self.dir_path / "img2.jpg")
        img = Image.new("RGB", (32, 32), color="red")
        img.save(self.img1_path)
        img.save(self.img2_path)

        # Non-existent image path
        self.fake_path = str(self.dir_path / "nonexistent_photo.png")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_extract_from_empty(self):
        """extract_images should return empty list on empty or None input."""
        self.assertEqual(extract_images(None), [])
        self.assertEqual(extract_images(""), [])
        self.assertEqual(extract_images([]), [])
        self.assertEqual(extract_images({}), [])

    def test_extract_multimodal_list(self):
        """Extracts paths from multimodal dicts: image_url, image, and path."""
        payload = [
            {"type": "text", "text": "Please inspect these attachments:"},
            {"type": "image_url", "image_url": {"url": self.img1_path}},
            {"type": "image", "image": self.img2_path},
            {"type": "image", "path": self.img1_path},
            {"type": "image_url", "image_url": f"file://{self.img2_path}"},
        ]
        results = extract_images(payload)
        # Should contain both img1 and img2, deduplicated
        self.assertEqual(len(results), 2)
        self.assertIn(str(Path(self.img1_path).resolve()), results)
        self.assertIn(str(Path(self.img2_path).resolve()), results)

    def test_extract_string_patterns(self):
        """Extracts [Image attached at: ...] and [Image: ...] patterns."""
        text = (
            f"Here is attachment 1: [Image attached at: {self.img1_path}]\n"
            f"And attachment 2: [Image: {self.img2_path}]"
        )
        results = extract_images(text)
        self.assertEqual(len(results), 2)
        self.assertIn(str(Path(self.img1_path).resolve()), results)
        self.assertIn(str(Path(self.img2_path).resolve()), results)

    def test_extract_explicit_file_paths_disk_check(self):
        """Explicit file paths ending in image extensions must exist on disk to be extracted."""
        text = (
            f"Please take a look at '{self.img1_path}' and also check {self.fake_path}."
        )
        results = extract_images(text)
        # Only existing file should be extracted; fake_path must be omitted
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], str(Path(self.img1_path).resolve()))

    def test_extract_deduplication(self):
        """Deduplicates paths across multiple mentions and formats."""
        text = (
            f"[Image attached at: {self.img1_path}] "
            f"[Image: {self.img1_path}] "
            f"Check out '{self.img1_path}'."
        )
        results = extract_images(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], str(Path(self.img1_path).resolve()))


class TestPluginLlmHooks(unittest.TestCase):
    """Test pre_llm_call and post_llm_call hooks."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

        # Isolated SQLite database for visual memory store
        self.db_path = str(self.dir_path / "test_memory.db")
        self.store = ImageMemoryStore(db_path=self.db_path)
        set_store(self.store)

        # Image A: White canvas with black square
        self.img_a_path = str(self.dir_path / "img_a.png")
        im_a = Image.new("RGB", (64, 64), color="white")
        d_a = ImageDraw.Draw(im_a)
        d_a.rectangle([10, 10, 54, 54], fill="black")
        im_a.save(self.img_a_path)

        # Image B: Distinct image (Black canvas with white circle)
        self.img_b_path = str(self.dir_path / "img_b.png")
        im_b = Image.new("RGB", (64, 64), color="black")
        d_b = ImageDraw.Draw(im_b)
        d_b.ellipse([10, 10, 54, 54], fill="white")
        im_b.save(self.img_b_path)

    def tearDown(self):
        set_store(None)
        self.store.close()
        self.test_dir.cleanup()

    def test_pre_llm_call_no_images(self):
        """pre_llm_call returns None when message contains no images."""
        result = pre_llm_call(user_message="Hello Hermes, what is the weather today?")
        self.assertIsNone(result)

    def test_pre_llm_call_different_images(self):
        """
        pre_llm_call with new different images saves each as a new record
        and returns None when no prior matches exist.
        """
        # Turn 1: Send Image A for the first time
        result_a = pre_llm_call(
            user_message=f"[Image attached at: {self.img_a_path}]",
            session_id="session-1",
        )
        self.assertIsNone(result_a, "First-time image should return None as no match exists")
        stats = self.store.get_stats()
        self.assertEqual(stats["total_images"], 1)

        # Turn 2: Send Image B (visually distinct)
        result_b = pre_llm_call(
            user_message=f"[Image: {self.img_b_path}]",
            session_id="session-2",
        )
        self.assertIsNone(result_b, "Distinct new image should return None")
        stats = self.store.get_stats()
        self.assertEqual(stats["total_images"], 2)

    def test_pre_llm_call_identical_image_hit(self):
        """
        pre_llm_call with an identical image recalls the memory,
        formats the rich injection context, and updates the hit count.
        """
        # Step 1: Record Image A initially
        pre_llm_call(
            user_message=f"[Image attached at: {self.img_a_path}]",
            session_id="session-init",
        )
        # Give Image A a summary
        matches = self.store.find_similar(self.img_a_path, threshold=0)
        self.assertEqual(len(matches), 1)
        record_id = matches[0]["id"]
        self.store.update_summary(record_id, summary="A high-contrast black square centered on white.")

        # Step 2: Send Image A a second time (identical image recall)
        res = pre_llm_call(
            user_message=f"Take a look at this again: [Image attached at: {self.img_a_path}]",
            session_id="session-repeat",
        )
        self.assertIsNotNone(res)
        self.assertIn("context", res)
        context = res["context"]

        # Verify formatting template
        self.assertIn("[视觉记忆库命中: 该图片与历史图片高度相似", context)
        self.assertIn("汉明距离 0", context)
        self.assertIn("累计发送 2 次", context)
        self.assertIn("历史上下文/AI记录: A high-contrast black square centered on white.", context)

        # Verify storage hit count was updated
        updated_rec = self.store.get_by_id(record_id)
        self.assertEqual(updated_rec["hit_count"], 2)

        # Step 3: Send Image A a third time
        res3 = pre_llm_call(
            user_message=f"[Image: {self.img_a_path}]",
            session_id="session-repeat-3",
        )
        self.assertIsNotNone(res3)
        self.assertIn("累计发送 3 次", res3["context"])
        updated_rec3 = self.store.get_by_id(record_id)
        self.assertEqual(updated_rec3["hit_count"], 3)

    def test_post_llm_call_summary_updating(self):
        """
        post_llm_call extracts a summary from assistant_response and updates empty
        summary in storage for the image, without overwriting existing summary.
        """
        # 1. Image A is registered via pre_llm_call with empty summary
        pre_llm_call(user_message=f"[Image attached at: {self.img_a_path}]")
        rec = self.store.find_similar(self.img_a_path, threshold=0)[0]
        self.assertIsNone(rec.get("ai_summary"))

        # 2. Assistant answers with visual analysis
        assistant_reply = (
            "This image portrays a crisp black square centered on a pure white background.\n\n"
            "The geometry is minimalist and sharp with no gradients."
        )
        post_llm_call(
            user_message=f"[Image attached at: {self.img_a_path}]",
            assistant_response=assistant_reply,
            session_id="session-1",
        )

        # 3. Verify storage was updated with first paragraph
        updated_rec = self.store.get_by_id(rec["id"])
        expected_summary = "This image portrays a crisp black square centered on a pure white background."
        self.assertEqual(updated_rec.get("ai_summary"), expected_summary)

        # 4. A subsequent turn with the same image should stack the new description onto existing summary
        secondary_reply = "Overwriting attempt summary."
        post_llm_call(
            user_message=f"[Image attached at: {self.img_a_path}]",
            assistant_response=secondary_reply,
            session_id="session-2",
        )
        persisted_rec = self.store.get_by_id(rec["id"])
        expected_stacked = f"{expected_summary}；Overwriting attempt summary."
        self.assertEqual(persisted_rec.get("ai_summary"), expected_stacked)

    def test_post_llm_call_truncation(self):
        """post_llm_call truncates long responses to 200 characters max."""
        pre_llm_call(user_message=f"[Image attached at: {self.img_a_path}]")
        rec = self.store.find_similar(self.img_a_path, threshold=0)[0]

        long_paragraph = "A" * 350
        post_llm_call(
            user_message=f"[Image attached at: {self.img_a_path}]",
            assistant_response=long_paragraph,
        )

        updated_rec = self.store.get_by_id(rec["id"])
        summary = updated_rec.get("ai_summary")
        self.assertIsNotNone(summary)
        self.assertEqual(len(summary), 200)
        self.assertEqual(summary, "A" * 200)

    def test_composite_dual_gating_behavior(self):
        """
        Verify Scheme B composite dual-gating:
        Matches only when dHash <= 24 AND pHash <= 12.
        If dHash <= 24 but pHash > 12, it must NOT match.
        """
        cad_entry = {
            "dhash": "00e0332dec238300",
            "phash": "ce91a96656e93d90",
            "summary": "CAD diagram of cart chassis",
        }
        self.store.record_image(
            cad_entry,
            summary=cad_entry["summary"],
            threshold=0,
        )

        # Frieren cosplay image fingerprint vs CAD:
        # dhash dist = 25 (> 24) and phash dist = 26 (> 12)
        frieren_fp = {
            "dhash": "020808cff470d3a1",
            "phash": "c4b3691da0eb569a",
        }
        matches = self.store.find_similar(
            frieren_fp,
            threshold=24,
            phash_threshold=12,
            require_both=True,
        )
        self.assertEqual(len(matches), 0, "Frieren cosplay must NOT match CAD diagram")

        # Case where dhash is within 24, but phash > 12 (different content, accidental gradient collision)
        fake_fp = {
            "dhash": "00e0332dec23830f",  # dhash dist 4 (<= 24)
            "phash": "ffffffffffffffff",  # phash dist > 12
        }
        matches_fake = self.store.find_similar(
            fake_fp,
            threshold=24,
            phash_threshold=12,
            require_both=True,
        )
        self.assertEqual(len(matches_fake), 0, "Discrepant phash must be blocked even if dhash is close")

        # Case where both dhash <= 24 and phash <= 12
        good_fp = {
            "dhash": "00e0332dec238303",  # dist 2
            "phash": "ce91a96656e93d93",  # dist 2
        }
        matches_good = self.store.find_similar(
            good_fp,
            threshold=24,
            phash_threshold=12,
            require_both=True,
        )
        self.assertEqual(len(matches_good), 1, "Should match when both dhash <= 24 and phash <= 12")

    def test_pre_and_post_llm_call_fail_open(self):
        """Both hooks should fail open and never raise exceptions on errors."""
        # pre_llm_call with invalid parameters
        res = pre_llm_call(user_message={"bad_structure": object()})
        self.assertIsNone(res)

        # post_llm_call with empty response or bad input
        try:
            post_llm_call(user_message=None, assistant_response="")
            post_llm_call(user_message="no images", assistant_response="some text")
        except Exception as exc:
            self.fail(f"post_llm_call raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
