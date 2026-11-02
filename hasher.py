"""
hasher.py - Pure-Python Perceptual Image Hashing using Pillow (PIL).

Provides robust, lightweight perceptual hashing algorithms:
1. Difference Hash (dHash) - Horizontal gradient differences (9x8).
2. Average Hash (aHash) - Luminance compared to mean (8x8).
3. Perceptual DCT Hash (pHash) - Pure Python 1D/2D DCT-II calculation without scipy/cv2 (32x32 -> 8x8).
4. Composite image fingerprint dictionary extraction.
5. Hamming distance & similarity scoring.

Handles transparency (RGBA/LA converted to white background RGB),
animated GIF/WEBP (uses frame 0), large image fast prescaling,
and invalid/corrupted files returning None gracefully.
"""

from __future__ import annotations

import functools
import io
import math
import os
import random
import statistics
import tempfile
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import unittest

from PIL import Image, ImageDraw, ImageOps

__all__ = [
    "compute_dhash",
    "compute_ahash",
    "compute_phash",
    "compute_image_fingerprint",
    "hamming_distance",
    "similarity_score",
]


@functools.lru_cache(maxsize=16)
def _get_dct_cosine_matrix(n: int, m: int) -> Tuple[Tuple[float, ...], ...]:
    """
    Precompute cosine basis matrix for 1D DCT-II:
    C[k][i] = cos(pi * (2 * i + 1) * k / (2 * n))
    where k is the frequency index (0 .. m-1)
    and i is the spatial sample index (0 .. n-1).
    """
    return tuple(
        tuple(math.cos(math.pi * (2 * i + 1) * k / (2.0 * n)) for i in range(n))
        for k in range(m)
    )


def _compute_dct_2d_lowfreq(
    matrix: Sequence[Sequence[float]],
    n: int,
    m: int,
) -> list[list[float]]:
    """
    Compute top-left m x m low-frequency coefficients of a 2D DCT-II
    on an n x n spatial matrix using pure Python separable 1D DCTs.

    Separable 2D DCT:
      1. 1D DCT on rows: n rows x n samples -> n rows x m frequencies.
      2. 1D DCT on columns: n rows x m frequencies -> m frequencies x m frequencies.
    """
    C = _get_dct_cosine_matrix(n, m)

    # Step 1: 1D DCT along rows (matrix is n x n, output row_dct is n x m)
    row_dct = [
        [sum(matrix[y][x] * C[u][x] for x in range(n)) for u in range(m)]
        for y in range(n)
    ]

    # Step 2: 1D DCT along columns of row_dct (output dct_2d is m x m)
    dct_2d = [
        [sum(row_dct[y][u] * C[v][y] for y in range(n)) for u in range(m)]
        for v in range(m)
    ]

    return dct_2d


def _bits_to_hex(bits: Sequence[bool]) -> str:
    """
    Pack a sequence of boolean bits into a zero-padded hex string (big-endian).
    For a 64-bit sequence, returns an exact 16-character hex string.
    """
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    hex_len = (len(bits) + 3) // 4
    return f"{val:0{hex_len}x}"


def _process_raw_image(
    img: Image.Image,
    file_size_bytes: Optional[int] = None,
) -> Optional[Tuple[Image.Image, int, int, Optional[int]]]:
    """
    Internal helper to validate, normalize, and preprocess a PIL Image.
    - Selects frame 0 for animated images (GIF/WEBP).
    - Applies EXIF orientation transpose if present.
    - Flattens transparency (RGBA/LA/PA/palette transparency) onto a white background RGB.
    - Decodes/loads pixel data to catch corrupted streams early.
    - Applies fast box-average prescaling if image is extremely large (> 1024px).

    Returns:
      (rgb_image, original_width, original_height, file_size_bytes)
      or None if decoding fails or dimensions are invalid.
    """
    try:
        # Rewind animated images to frame 0
        if hasattr(img, "seek"):
            try:
                img.seek(0)
            except Exception:
                pass

        # Handle EXIF rotation if present
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        orig_w, orig_h = img.size
        if orig_w <= 0 or orig_h <= 0:
            return None

        # Flatten transparency onto solid white RGB
        if img.mode in ("RGBA", "LA", "PA", "RGBa"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(bg, rgba)
            rgb_img = composite.convert("RGB")
        elif img.mode == "P" and "transparency" in img.info:
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(bg, rgba)
            rgb_img = composite.convert("RGB")
        elif img.mode != "RGB":
            rgb_img = img.convert("RGB")
        else:
            rgb_img = img.copy()

        # Force decode to catch corrupted or truncated image streams
        rgb_img.load()

        # Fast prescale for ultra-large images to ensure single-image hashing < 5ms
        max_dim = max(rgb_img.size)
        if max_dim > 1024:
            factor = max(1, max_dim // 512)
            rgb_img = rgb_img.reduce(factor)

        return rgb_img, orig_w, orig_h, file_size_bytes

    except Exception:
        return None


def _prepare_image(
    img_or_path: Union[str, os.PathLike, Image.Image, io.BytesIO, bytes],
) -> Optional[Tuple[Image.Image, int, int, Optional[int]]]:
    """
    Load and preprocess an image from a path, raw bytes, BytesIO, or PIL Image.
    Ensures file descriptors are safely closed.
    Returns (rgb_image, original_width, original_height, file_size_bytes) or None on failure.
    """
    if img_or_path is None:
        return None

    try:
        if isinstance(img_or_path, (str, os.PathLike)):
            if not os.path.isfile(img_or_path):
                return None
            file_size_bytes = os.path.getsize(img_or_path)
            with Image.open(img_or_path) as opened_img:
                return _process_raw_image(opened_img, file_size_bytes=file_size_bytes)

        elif isinstance(img_or_path, (bytes, io.BytesIO)):
            buf = io.BytesIO(img_or_path) if isinstance(img_or_path, bytes) else img_or_path
            file_size_bytes = len(buf.getbuffer()) if hasattr(buf, "getbuffer") else None
            with Image.open(buf) as opened_img:
                return _process_raw_image(opened_img, file_size_bytes=file_size_bytes)

        elif isinstance(img_or_path, Image.Image):
            file_size_bytes = None
            filename = getattr(img_or_path, "filename", None)
            if filename and isinstance(filename, (str, os.PathLike)) and os.path.isfile(filename):
                try:
                    file_size_bytes = os.path.getsize(filename)
                except Exception:
                    file_size_bytes = None
            return _process_raw_image(img_or_path, file_size_bytes=file_size_bytes)

        else:
            return None

    except Exception:
        return None


def _compute_dhash_from_prepared(img: Image.Image, hash_size: int = 8) -> str:
    """Compute difference hash directly from a preprocessed RGB image."""
    # Resize to (width = hash_size + 1, height = hash_size)
    resized = img.convert("L").resize(
        (hash_size + 1, hash_size),
        resample=Image.Resampling.LANCZOS,
    )
    pixels = list(resized.getdata())
    width = hash_size + 1
    bits = []
    for y in range(hash_size):
        row_offset = y * width
        for x in range(hash_size):
            bits.append(pixels[row_offset + x + 1] > pixels[row_offset + x])
    return _bits_to_hex(bits)


def _compute_ahash_from_prepared(img: Image.Image, hash_size: int = 8) -> str:
    """Compute average hash directly from a preprocessed RGB image."""
    # Resize to (width = hash_size, height = hash_size)
    resized = img.convert("L").resize(
        (hash_size, hash_size),
        resample=Image.Resampling.LANCZOS,
    )
    pixels = list(resized.getdata())
    avg = sum(pixels) / len(pixels)
    bits = [p > avg for p in pixels]
    return _bits_to_hex(bits)


def _compute_phash_from_prepared(
    img: Image.Image,
    hash_size: int = 8,
    highfreq_factor: int = 4,
) -> str:
    """Compute DCT perceptual hash directly from a preprocessed RGB image."""
    n = hash_size * highfreq_factor
    m = hash_size
    resized = img.convert("L").resize(
        (n, n),
        resample=Image.Resampling.LANCZOS,
    )
    pixels = list(resized.getdata())
    matrix = [pixels[y * n : (y + 1) * n] for y in range(n)]

    # Compute low-frequency m x m DCT coefficients
    dct_2d = _compute_dct_2d_lowfreq(matrix, n=n, m=m)

    # Flatten in row-major order
    lowfreq = [dct_2d[v][u] for v in range(m) for u in range(m)]
    med = statistics.median(lowfreq)
    bits = [val > med for val in lowfreq]
    return _bits_to_hex(bits)


def compute_dhash(
    img_or_path: Union[str, os.PathLike, Image.Image, io.BytesIO, bytes],
    hash_size: int = 8,
) -> Optional[str]:
    """
    Compute Difference Hash (dHash) returning a 16-character hex string (for hash_size=8).
    Returns None if the image cannot be read or is invalid/corrupted.
    """
    if hash_size < 2:
        raise ValueError("hash_size must be >= 2")
    prep = _prepare_image(img_or_path)
    if prep is None:
        return None
    try:
        return _compute_dhash_from_prepared(prep[0], hash_size=hash_size)
    except Exception:
        return None


def compute_ahash(
    img_or_path: Union[str, os.PathLike, Image.Image, io.BytesIO, bytes],
    hash_size: int = 8,
) -> Optional[str]:
    """
    Compute Average Hash (aHash) returning a 16-character hex string (for hash_size=8).
    Returns None if the image cannot be read or is invalid/corrupted.
    """
    if hash_size < 2:
        raise ValueError("hash_size must be >= 2")
    prep = _prepare_image(img_or_path)
    if prep is None:
        return None
    try:
        return _compute_ahash_from_prepared(prep[0], hash_size=hash_size)
    except Exception:
        return None


def compute_phash(
    img_or_path: Union[str, os.PathLike, Image.Image, io.BytesIO, bytes],
    hash_size: int = 8,
    highfreq_factor: int = 4,
) -> Optional[str]:
    """
    Compute Perceptual DCT Hash (pHash) returning a 16-character hex string (for hash_size=8).
    Uses a pure Python 1D/2D DCT-II calculation without scipy/cv2.
    Returns None if the image cannot be read or is invalid/corrupted.
    """
    if hash_size < 2:
        raise ValueError("hash_size must be >= 2")
    prep = _prepare_image(img_or_path)
    if prep is None:
        return None
    try:
        return _compute_phash_from_prepared(
            prep[0],
            hash_size=hash_size,
            highfreq_factor=highfreq_factor,
        )
    except Exception:
        return None


def compute_image_fingerprint(
    img_or_path: Union[str, os.PathLike, Image.Image, io.BytesIO, bytes],
) -> Optional[Dict[str, Any]]:
    """
    Compute composite perceptual fingerprint dictionary for an image:
    {
        'dhash': 16-char hex string,
        'ahash': 16-char hex string,
        'phash': 16-char hex string,
        'width': original image width (int),
        'height': original image height (int),
        'file_size_bytes': file size on disk or in-memory size (int or None),
    }
    Returns None gracefully if the image is invalid or corrupted.
    """
    prep = _prepare_image(img_or_path)
    if prep is None:
        return None

    rgb_img, orig_w, orig_h, file_size_bytes = prep
    try:
        dh = _compute_dhash_from_prepared(rgb_img, hash_size=8)
        ah = _compute_ahash_from_prepared(rgb_img, hash_size=8)
        ph = _compute_phash_from_prepared(rgb_img, hash_size=8, highfreq_factor=4)
        return {
            "dhash": dh,
            "ahash": ah,
            "phash": ph,
            "width": orig_w,
            "height": orig_h,
            "file_size_bytes": file_size_bytes,
        }
    except Exception:
        return None


def hamming_distance(hex1: str, hex2: str) -> int:
    """
    Calculate the Hamming distance (number of bit differences) between two hex strings.
    """
    if not isinstance(hex1, str) or not isinstance(hex2, str):
        raise TypeError(f"Hash values must be strings, got {type(hex1).__name__} and {type(hex2).__name__}")

    clean1 = hex1.strip()
    clean2 = hex2.strip()
    if not clean1 or not clean2:
        raise ValueError("Hash strings cannot be empty")

    try:
        val1 = int(clean1, 16)
        val2 = int(clean2, 16)
    except ValueError as e:
        raise ValueError(f"Invalid hexadecimal string: {e}") from e

    return (val1 ^ val2).bit_count()


def similarity_score(hex1: str, hex2: str, total_bits: int = 64) -> float:
    """
    Calculate similarity score between two hex hash strings from 0.0 to 1.0.
    1.0 means identical hashes (0 bit difference), 0.0 means completely different.
    """
    if total_bits <= 0:
        raise ValueError("total_bits must be positive")
    dist = hamming_distance(hex1, hex2)
    score = 1.0 - (dist / float(total_bits))
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
class TestPerceptualHasher(unittest.TestCase):
    """Unit tests for perceptual image hashing, similarity scoring, and robustness."""

    def setUp(self):
        # Create a rich baseline synthetic image (300x300) with various features
        self.base_img = Image.new("RGB", (300, 300), color=(245, 245, 245))
        draw = ImageDraw.Draw(self.base_img)
        draw.rectangle([25, 25, 120, 120], fill=(30, 80, 200))
        draw.ellipse([100, 100, 260, 260], fill=(220, 60, 60))
        draw.line([0, 300, 300, 0], fill=(40, 190, 40), width=6)
        draw.polygon([(150, 20), (280, 80), (220, 160)], fill=(240, 200, 30))

    def test_identical_images(self):
        """Identical images must yield identical hashes (distance = 0, similarity = 1.0)."""
        copy_img = self.base_img.copy()

        dh1 = compute_dhash(self.base_img)
        dh2 = compute_dhash(copy_img)
        self.assertIsNotNone(dh1)
        self.assertIsNotNone(dh2)
        self.assertEqual(len(dh1), 16)
        self.assertEqual(dh1, dh2)
        self.assertEqual(hamming_distance(dh1, dh2), 0)
        self.assertEqual(similarity_score(dh1, dh2), 1.0)

        ah1 = compute_ahash(self.base_img)
        ah2 = compute_ahash(copy_img)
        self.assertIsNotNone(ah1)
        self.assertIsNotNone(ah2)
        self.assertEqual(len(ah1), 16)
        self.assertEqual(ah1, ah2)
        self.assertEqual(hamming_distance(ah1, ah2), 0)
        self.assertEqual(similarity_score(ah1, ah2), 1.0)

        ph1 = compute_phash(self.base_img)
        ph2 = compute_phash(copy_img)
        self.assertIsNotNone(ph1)
        self.assertIsNotNone(ph2)
        self.assertEqual(len(ph1), 16)
        self.assertEqual(ph1, ph2)
        self.assertEqual(hamming_distance(ph1, ph2), 0)
        self.assertEqual(similarity_score(ph1, ph2), 1.0)

        fp1 = compute_image_fingerprint(self.base_img)
        fp2 = compute_image_fingerprint(copy_img)
        self.assertIsNotNone(fp1)
        self.assertIsNotNone(fp2)
        self.assertEqual(fp1["dhash"], fp2["dhash"])
        self.assertEqual(fp1["ahash"], fp2["ahash"])
        self.assertEqual(fp1["phash"], fp2["phash"])
        self.assertEqual(fp1["width"], 300)
        self.assertEqual(fp1["height"], 300)

    def test_resized_images(self):
        """Resized images should retain high similarity and small hamming distance."""
        # Downscaled to 150x150
        downscaled = self.base_img.resize((150, 150), Image.Resampling.BILINEAR)
        # Upscaled to 450x450
        upscaled = self.base_img.resize((450, 450), Image.Resampling.BICUBIC)

        for target in (downscaled, upscaled):
            for hash_fn, name in [
                (compute_dhash, "dhash"),
                (compute_ahash, "ahash"),
                (compute_phash, "phash"),
            ]:
                h_orig = hash_fn(self.base_img)
                h_target = hash_fn(target)
                dist = hamming_distance(h_orig, h_target)
                sim = similarity_score(h_orig, h_target)
                # Resizing should produce very small distance (<= 4 bits out of 64)
                self.assertLessEqual(
                    dist,
                    4,
                    f"{name} distance too large for resized image: {dist}",
                )
                self.assertGreaterEqual(
                    sim,
                    0.90,
                    f"{name} similarity too low for resized image: {sim}",
                )

    def test_noise_images(self):
        """Images with minor noise should have low distance; completely different images high distance."""
        # Add moderate random pixel noise (+/- 12 jitter)
        noisy = self.base_img.copy()
        px = noisy.load()
        rng = random.Random(42)
        w, h = noisy.size
        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y]
                j = rng.randint(-12, 12)
                px[x, y] = (
                    max(0, min(255, r + j)),
                    max(0, min(255, g + j)),
                    max(0, min(255, b + j)),
                )

        for hash_fn, name in [
            (compute_dhash, "dhash"),
            (compute_ahash, "ahash"),
            (compute_phash, "phash"),
        ]:
            h_orig = hash_fn(self.base_img)
            h_noisy = hash_fn(noisy)
            dist = hamming_distance(h_orig, h_noisy)
            sim = similarity_score(h_orig, h_noisy)
            # Mild noise distance should be <= 8 bits
            self.assertLessEqual(
                dist,
                8,
                f"{name} distance too large for noisy image: {dist}",
            )
            self.assertGreaterEqual(sim, 0.85)

        # Create a completely different image (inverted colors and alternate geometry)
        diff_img = Image.new("RGB", (300, 300), color=(15, 20, 35))
        diff_draw = ImageDraw.Draw(diff_img)
        diff_draw.rectangle([180, 180, 290, 290], fill=(255, 240, 10))
        diff_draw.ellipse([10, 150, 120, 280], fill=(240, 50, 200))

        for hash_fn, name in [
            (compute_dhash, "dhash"),
            (compute_ahash, "ahash"),
            (compute_phash, "phash"),
        ]:
            h_orig = hash_fn(self.base_img)
            h_diff = hash_fn(diff_img)
            dist = hamming_distance(h_orig, h_diff)
            sim = similarity_score(h_orig, h_diff)
            # Unrelated images should have distance around 20-40 (typical random distance is ~32)
            self.assertGreaterEqual(
                dist,
                18,
                f"{name} distance too small for unrelated image: {dist}",
            )
            self.assertLessEqual(sim, 0.75)

    def test_transparency_handling(self):
        """RGBA/LA transparent backgrounds must be converted onto white background RGB."""
        # Create RGBA image with transparent background and black circle
        rgba = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        draw_rgba = ImageDraw.Draw(rgba)
        draw_rgba.ellipse([20, 20, 100, 100], fill=(0, 0, 0, 255))

        # Create equivalent RGB image directly on white background
        rgb = Image.new("RGB", (120, 120), (255, 255, 255))
        draw_rgb = ImageDraw.Draw(rgb)
        draw_rgb.ellipse([20, 20, 100, 100], fill=(0, 0, 0))

        for hash_fn in (compute_dhash, compute_ahash, compute_phash):
            h_rgba = hash_fn(rgba)
            h_rgb = hash_fn(rgb)
            self.assertIsNotNone(h_rgba)
            self.assertIsNotNone(h_rgb)
            self.assertEqual(h_rgba, h_rgb)

        # Test LA mode
        la = Image.new("LA", (120, 120), (0, 0))
        draw_la = ImageDraw.Draw(la)
        draw_la.ellipse([20, 20, 100, 100], fill=(0, 255))
        for hash_fn in (compute_dhash, compute_ahash, compute_phash):
            h_la = hash_fn(la)
            h_rgb = hash_fn(rgb)
            self.assertEqual(h_la, h_rgb)

    def test_animated_gif(self):
        """Animated GIF should reliably use frame 0."""
        f0 = Image.new("RGB", (100, 100), (255, 0, 0))
        f1 = Image.new("RGB", (100, 100), (0, 0, 255))

        buf = io.BytesIO()
        f0.save(
            buf,
            format="GIF",
            save_all=True,
            append_images=[f1],
            duration=100,
            loop=0,
        )
        gif_bytes = buf.getvalue()

        for hash_fn in (compute_dhash, compute_ahash, compute_phash):
            h_gif = hash_fn(gif_bytes)
            h_f0 = hash_fn(f0)
            self.assertEqual(h_gif, h_f0)

    def test_corrupted_and_invalid_files(self):
        """Invalid, missing, or corrupted files must return None gracefully."""
        self.assertIsNone(compute_dhash("non_existent_file_path_12345.jpg"))
        self.assertIsNone(compute_ahash("non_existent_file_path_12345.jpg"))
        self.assertIsNone(compute_phash("non_existent_file_path_12345.jpg"))
        self.assertIsNone(compute_image_fingerprint("non_existent_file_path_12345.jpg"))

        # Corrupted image bytes
        corrupted = b"NOT AN IMAGE FILE AT ALL"
        self.assertIsNone(compute_dhash(corrupted))
        self.assertIsNone(compute_ahash(corrupted))
        self.assertIsNone(compute_phash(corrupted))
        self.assertIsNone(compute_image_fingerprint(corrupted))

        # Truncated PNG in temporary file
        png_buf = io.BytesIO()
        self.base_img.save(png_buf, format="PNG")
        truncated_bytes = png_buf.getvalue()[:40]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(truncated_bytes)
            truncated_path = tf.name

        try:
            self.assertIsNone(compute_dhash(truncated_path))
            self.assertIsNone(compute_ahash(truncated_path))
            self.assertIsNone(compute_phash(truncated_path))
            self.assertIsNone(compute_image_fingerprint(truncated_path))
        finally:
            if os.path.exists(truncated_path):
                os.remove(truncated_path)

        # None and invalid types
        self.assertIsNone(compute_dhash(None))
        self.assertIsNone(compute_image_fingerprint(None))
        self.assertIsNone(compute_dhash(12345))  # type: ignore

    def test_fingerprint_file(self):
        """compute_image_fingerprint should report file size and dimensions accurately."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            temp_path = tf.name
            self.base_img.save(temp_path, format="PNG")

        try:
            fp = compute_image_fingerprint(temp_path)
            self.assertIsNotNone(fp)
            self.assertEqual(fp["width"], 300)
            self.assertEqual(fp["height"], 300)
            self.assertEqual(fp["file_size_bytes"], os.path.getsize(temp_path))
            self.assertGreater(fp["file_size_bytes"], 0)
            self.assertEqual(len(fp["dhash"]), 16)
            self.assertEqual(len(fp["ahash"]), 16)
            self.assertEqual(len(fp["phash"]), 16)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_hamming_and_similarity_math(self):
        """Verify exact hamming distances, similarity score bounds, and exceptions."""
        self.assertEqual(hamming_distance("0000000000000000", "0000000000000000"), 0)
        self.assertEqual(similarity_score("0000000000000000", "0000000000000000"), 1.0)

        self.assertEqual(hamming_distance("0000000000000000", "0000000000000001"), 1)
        self.assertAlmostEqual(similarity_score("0000000000000000", "0000000000000001"), 63.0 / 64.0)

        self.assertEqual(hamming_distance("0000000000000000", "ffffffffffffffff"), 64)
        self.assertEqual(similarity_score("0000000000000000", "ffffffffffffffff"), 0.0)

        self.assertEqual(hamming_distance("ffff0000ffff0000", "0000000000000000"), 32)
        self.assertEqual(similarity_score("ffff0000ffff0000", "0000000000000000"), 0.5)

        # Error cases
        with self.assertRaises(TypeError):
            hamming_distance(None, "0000")  # type: ignore
        with self.assertRaises(ValueError):
            hamming_distance("", "0000")
        with self.assertRaises(ValueError):
            hamming_distance("xyzg", "0000")
        with self.assertRaises(ValueError):
            similarity_score("0000", "0000", total_bits=0)


if __name__ == "__main__":
    print("Running hasher.py perceptual hashing unit tests...")
    unittest.main()
