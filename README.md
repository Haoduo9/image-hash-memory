# Image Hash Memory for Hermes Agent

A plug-and-play visual memory and auto-recall plugin for Hermes Agent.

It solves multimodal LLMs' "visual amnesia" across session compactions, model switching, and restarts by indexing images using Perceptual Hashing (dHash + aHash + pHash) in a local lightweight SQLite store with zero token costs and sub-5ms retrieval latency.

---

## 🌟 Key Features
- **Zero LLM Token Overhead**: Uses pure-Python perceptual hashes (dHash + aHash + pHash) via Pillow.
- **Composite Dual-Gating (综合双门禁校验)**: Combines spatial difference gradient (`dHash <= 24`) and frequency DCT energy distribution (`pHash <= 12`) in a dual-voting gatekeeper, completely preventing false positives and accidental hash collisions between visually distinct images.
- **Dual-Layer Anti-Filter / Anti-Crop Matching**: 
  - Standard gradient difference hash (dHash) handles crop and aspect-ratio variations.
  - Discrete Cosine Transform low-frequency hash (pHash) resists extreme color filters, contrast shifts, and color grading.
- **Cross-Session Persistence**: Stores hashes and contextual dialogue in a standalone SQLite DB (`~/.hermes/data/image_memory.db`).
- **Seamless Context Injection**: Intercepts turns via `pre_llm_call` to automatically inject historical recognition notes when re-sent similar images are matched.
- **Self-Expanding Memory**: Merges follow-up discussions and corrections back into the original image's record with a configurable Hamming threshold (`<= 24`).
- **Strict Turn Isolation**: Only triggers visual recall when the current turn's user message actually contains image attachments.
- **Fail-Open Architecture**: Any edge-case parsing or matching failure fails open safely without interrupting chat turns.

---

## 📦 Installation

```bash
hermes plugins install https://github.com/Haoduo9/image-hash-memory.git
```

Or for local development:
```bash
git clone https://github.com/Haoduo9/image-hash-memory.git
hermes plugins install ./image-hash-memory
```

---

## ⚙️ Configuration
Edit `config.yaml` inside the plugin directory (hot-reloaded instantly without restarting the gateway):

```yaml
# Perceptual hash Hamming distance match threshold (default 24)
hamming_threshold: 24

# Enable pHash (DCT) dual insurance against heavy color grading
enable_phash: true
phash_threshold: 12

# Verification mode: composite (dual-gating requires both dHash and pHash to pass)
verification_mode: composite

# Master toggle for memory recall injection
enable_recall: true

# Auto-append new descriptions into existing records
auto_append_summary: true

# Max characters extracted per turn description
max_summary_length: 200

# SQLite storage path
db_path: "~/.hermes/data/image_memory.db"
```

---

## 🧪 Testing

```bash
python3 -m unittest -v test_plugin
```

To validate with the Hermes plugin manager:
```bash
hermes plugins validate /path/to/image-hash-memory
```

---

## 📄 License
MIT License. Copyright (c) 2026 Haoduo9.
