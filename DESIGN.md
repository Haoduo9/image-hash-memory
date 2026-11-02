# Image Hash Memory & Auto-Recall for Hermes

通用图片感知哈希（Perceptual Hash）记忆与自动召回插件系统。

## 架构规划

1. **hasher.py**
   - 纯 Python + Pillow (PIL)，零外部复杂 C 依赖，超低开销（单张图 < 5ms）。
   - 实现三种经典感知哈希并支持复合指纹：
     - `dhash` (Difference Hash): 9x8 梯度对比，抗等比缩放、压缩伪影。
     - `ahash` (Average Hash): 8x8 均值对比，捕捉整体光照结构。
     - `phash` (Perceptual DCT Hash): 32x32 DCT 离散余弦变换低频系数对比，抗轻度旋转、裁剪和色调微调。
   - 健壮性保障：支持透明度通道（RGBA/LA 自动转白底 RGB）、动图 GIF/WEBP 提取第 0 帧、超大图片先快速缩放、损坏图片异常捕获返回 None。
   - 汉明距离计算 `hamming_distance(hash1, hash2)` 与相似度百分比估算。

2. **storage.py**
   - SQLite 数据库支持（默认保存在 `~/.hermes/data/image_memory.db`）。
   - 表结构：
     - `images`:
       - `id INTEGER PRIMARY KEY AUTOINCREMENT`
       - `dhash TEXT NOT NULL`
       - `ahash TEXT`
       - `phash TEXT`
       - `file_path TEXT`
       - `file_size INTEGER`
       - `width INTEGER`
       - `height INTEGER`
       - `first_seen_at TIMESTAMP`
       - `last_seen_at TIMESTAMP`
       - `seen_count INTEGER DEFAULT 1`
       - `user_caption TEXT` (初次或每次附加的用户输入)
       - `ai_summary TEXT` (AI 关于该图的总结/分析)
       - `session_id TEXT`
       - `platform TEXT`
   - 查询方法：
     - `find_similar(image_path_or_hashes, threshold=8, limit=3)`: 扫描数据库，计算汉明距离，返回距离最近的候选结果列表。
     - `record_image(img_path, user_caption, ai_summary, session_id, platform, threshold=8)`:
       如果存在相似度在阈值内的旧图，则累加 `seen_count`、更新 `last_seen_at`，返回已存在记录；如果不存在，则插入新记录并返回新记录 ID。
     - `update_summary(record_id, ai_summary)`: 在对话结束时更新 AI 对该图片的分析摘要。

3. **plugin.py / __init__.py (Hermes Hook)**
   - 满足 Hermes 插件规范（`plugin.yaml` + `__init__.py`），注册 `pre_llm_call` 和 `post_llm_call`。
   - `pre_llm_call`:
     - 检查 `user_message`（无论是包含 `[Image attached at: ...]` 的纯字符串，还是包含 image_url 的多模态 List）。
     - 提取图片本地路径并校验存在性。
     - 调用 `storage.find_similar()`。若命中：
       - 拼装注入上下文：`[视觉记忆库命中: 该图片与历史图片高度相似（汉明距离 {d}，首次出现于 {first_seen}，累计发送 {count} 次）。历史上下文/AI记录: {ai_summary}]`
       - 返回 `{"context": injection_text}`。
     - 若未命中：将图片作为新图预登记入库，并记录当前 `turn_id`/`session_id`。
   - `post_llm_call`:
     - 提取本次 turn 的 `assistant_response`，如果本轮涉及新图，自动提取摘要更新到该图的 `ai_summary` 中。
   - Fail-Open 哲学：任何异常、超时或无法解析图片均静默记录警告，绝对不阻断用户正常对话。

4. **cli.py & tests**
   - 独立的命令行工具与自动化测试套件。
