# Image Hash Memory for Hermes Agent

A plug-and-play visual memory and auto-recall plugin for Hermes Agent.

It solves multimodal LLMs' "visual amnesia" across session compactions and restarts by indexing images using Perceptual Hashing (dHash + pHash) in a local lightweight SQLite store with zero token costs and sub-5ms retrieval latency.

## Features
- **Zero Token Overhead**: Uses pure-Python perceptual hashes (dHash + aHash + pHash) via Pillow.
- **Cross-Session Persistence**: Stores hashes and contextual dialogue in a standalone SQLite DB (`~/.hermes/data/image_memory.db`).
- **Seamless Context Injection**: Intercepts turns via `pre_llm_call` to automatically inject historical recognition notes when re-sent similar images are matched.
- **Self-Expanding Memory**: Merges follow-up discussions and corrections back into the original image's record with a configurable Hamming threshold.
- **Fail-Open Architecture**: Any edge-case parsing or matching failure fails open safely without interrupting chat turns.

## Installation
```bash
hermes plugins install /path/to/image-hash-memory
# or git URL
hermes plugins install https://github.com/<user>/image-hash-memory.git
```

## Configuration
Edit `config.yaml` inside the plugin directory (hot-reloaded instantly):
```yaml
hamming_threshold: 25
enable_phash: true
phash_threshold: 12
enable_recall: true
auto_append_summary: true
max_summary_length: 200
db_path: "~/.hermes/data/image_memory.db"
```
