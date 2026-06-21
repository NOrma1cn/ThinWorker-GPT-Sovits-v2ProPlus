# CHANGES.md — thin-tts-server 优化与魔改记录

本文档记录了从 GPT-SoVITS 原始代码到 thin-tts-server 独立包过程中的所有修改，包括裁剪、Bug 修复、性能优化，以及已尝试但不可行的方向。

## 打包裁剪

从完整 GPT-SoVITS 仓库精简为推理专用包，去掉了以下内容：

**功能模块：** 训练代码（training_step、configure_optimizers、losses、data_utils）、多版本分支（v1/v2/v3/v4/v2Pro，仅保留 v2ProPlus）、多语言 G2P（仅保留中文 chinese2）、BigVGAN vocoder、super_sampling 超分、音频后处理超采样、DPO 训练相关函数。

**架构调整：** 去掉 `os.chdir` / `sys.path` hack，所有 import 改为包内路径；`tools/i18n/i18n.py` 的 `I18nAuto` 简化为 passthrough stub（直接返回原文）；G2PW ONNX 模型改为延迟加载（避免 import 时阻塞）；`cleaner.py` 只保留中文分支。

**裁剪效果：** 从 ~150+ 源文件精简到 37 个文件，wheel 包大小 ~1.1MB。

## Bug 修复

### 1. TTS_Config 传参结构

**问题：** 服务器日志显示 "fall back to default t2s_weights_path"，传入的权重路径被忽略。

**根因：** `TTS_Config.__init__` 的解析逻辑是 `configs_.get("custom", configs_["v2ProPlus"])`——平铺的 dict 没有 `"custom"` key，也没有 `"v2ProPlus"` key，导致 fallback 到内置默认路径。

**修复：** 在 `server.py` 中将配置包装为 `{"custom": {...}}` 嵌套结构。

### 2. torchaudio.load 在 Windows 上不可用

**问题：** `ImportError: TorchCodec is required for load_with_torchcodec`。torchaudio 2.11 的 `load` 函数默认使用 torchcodec 后端，而 torchcodec 在 Windows 上不可用；且 2.11 版本已移除 soundfile 后端支持，`backend="soundfile"` 参数被忽略。

**修复：** 用 `soundfile.read()` + `torch.from_numpy()` 替代 `torchaudio.load`，处理维度转置（stereo → channels-first）。在 pyproject.toml 添加 `soundfile>=0.12` 依赖。

### 3. fast_langdetect 缓存目录不存在

**问题：** `FileNotFoundError: fast-langdetect: Cache directory not found: ...\thin_tts\pretrained_models\fast_langdetect`。`langsegmenter.py` 中自定义的缓存路径指向包内不存在的目录。

**修复：** 移除自定义缓存配置，让 fast_langdetect 使用默认缓存位置（`~/.cache/fast_langdetect`）。可通过环境变量 `THIN_TTS_LANGDETECT_CACHE` 自定义。

### 4. audio_postprocess 参数残留

**问题：** `TypeError: TTS.audio_postprocess() takes from 3 to 7 positional arguments but 8 were given`。裁剪 `super_sampling` 参数后，3 处调用点遗漏了末尾的 `False` 参数。

**修复：** 逐一修正 3 处调用（streaming 首包 yield、streaming 中间 chunk yield、non-streaming final yield），去掉多余的 `False` 参数。

### 5. i18n relpath 跨盘符崩溃

**问题：** `tools/i18n/i18n.py` 中 `os.path.relpath` 在 CWD 与文件所在路径跨盘符（如 C: vs D:）时抛出 `ValueError`。

**修复：** 改用 `os.path.abspath(__file__)` 获取绝对路径，避免依赖 CWD。最终简化为 passthrough stub。

### 6. pip 重装不更新新增文件

**问题：** `pip install --force-reinstall --no-deps` 不会删除新增的文件（不在 RECORD 中），导致旧代码残留。

**应对：** 先 `pip uninstall -y`，再手动删除 `site-packages/thin_tts` 目录，最后重新安装。

## 性能优化（已验证有效）

### 去掉 @torch.jit.script

**效果：** AR 推理速度 50 → 100 it/s（2x 提升）。

**原因：** 小 batch（batch=1）逐 token 解码场景下，JIT 编译的 graph 优化开销大于收益。`@torch.jit.script` 装饰在 `infer_panel_naive` 的 `decode_next_token` 上，每步都有额外的 type check 和 dispatch 开销。

### SV embedding 缓存

**效果：** 相同参考音频的请求省去 ~200ms。

**原理：** Speaker Verification 模型（ERes2NetV2）对参考音频提取 embedding 后缓存，同一参考音频的后续请求直接复用。首包延迟从 ~380ms 降至 ~278ms（SV 缓存命中时）。

### thin streaming server 架构

**效果：** 中间代理开销从 285ms 降至 ~10ms。

**设计：** front worker 通过 aiohttp `ClientSession` + `iter_any()` 直接 pipe thin server 的流式响应，避免传统 HTTP client 的缓冲行为。`ClientSession` 在应用启动时创建（`TCPConnector limit=10`），避免 per-request TCP 握手。

### VITS decode_streaming ge 缓存

**效果：** 首包延迟 -39%（chunk=12 时 0.310s → 0.190s）。

**原理：** VITS 解码器中 `ge`（global encoding）在每个 chunk 的 `decode_streaming` 中被重复计算，但 `ge` 只依赖参考 spec，与当前 chunk 无关。缓存后每个 chunk 只计算一次。

### chunk_length 调优

**效果：** chunk=10 vs chunk=16，15 字文本首包 247ms → 190ms（-23%），36 字文本 468ms → 179ms（-62%），43 字文本 489ms → 348ms（-29%）。

**原理：** `chunk_length` 控制 VITS streaming 解码每多少个 token yield 一次音频。越小的 chunk 凑够 yield 阈值越快，首包延迟越低；但 VITS 重编码次数增多，总耗时基本持平。

### 去掉 torch.jit.script（详细）

AR 模型的 `infer_panel_naive` 是生产推理路径，内部调用 `decode_next_token`。在 GPT-SoVITS 原版中 `decode_next_token` 被 `@torch.jit.script` 装饰。实测在 batch=1、逐 token 解码的小 batch 场景下，JIT 反而引入额外开销（type check、graph dispatch），去掉后速度翻倍。

## 实测性能数据

测试环境：NVIDIA GPU、CUDA、Windows 10/11、Python 3.11、hutao v2ProPlus 模型。

### 流式推理延迟（thin server 直连）

| 文本长度 | 首包延迟 | 总耗时 | 音频时长 | chunks |
|----------|----------|--------|----------|--------|
| 15 字 | 190-220ms | 1.28s | 3.54s | 10 |
| 36 字 | ~180-430ms | — | — | — |
| 46 字（首次，无 SV 缓存） | 629ms | 666ms | — | — |
| 46 字（SV 缓存命中） | 421ms | 458ms | — | — |

### 端到端延迟（front worker → thin server）

| 场景 | 端到端延迟 | TTS 生成耗时 |
|------|-----------|-------------|
| seq=0（首次，无 SV 缓存） | 666ms | 629ms |
| seq=1（SV 缓存命中） | 458ms | 421ms |

### Stress Test（cut0，200 句）

18 句 LONG（>10s）/ 0 ERROR。5 条长句文本每次复现都 LONG（时长完全一致）——EOS 未及时触发的 overlong 异常，非吞句。

## 已尝试但不可行的方向

### torch.compile（Windows）

- **inductor 后端：** 需要 Triton，Windows 上不可用 → `TritonMissing` 错误
- **cudagraphs 后端：** AR 模型的 `self._seq_len` 每个 token 变化，导致 CUDA graph 不断重编译；命中 `recompile_limit`（8）后回退 eager，首包 0.242s（比无 compile 的 0.206s 更慢），AR 速度 67 it/s（比无 compile 的 100 it/s 更慢）
- **结论：** Windows 环境下 torch.compile 当前不可行

### CUDA Graph streaming

- **方案：** 只捕获单步 decode（12 层 forward）为 CUDA Graph，streaming yield 在 Python 控制循环中执行
- **问题：** 短文本场景下 capture 开销 +59%，得不偿失
- **需要的改造：** KV cache 全 buffer + attn mask（替代动态切片）、PE 在 Graph 外计算 copy 到静态 buffer、`_handle_request` 改 generator
- **现有代码：** `t2s_model_cudagraph.py` 的 `capture()` 已实现核心逻辑，但只接入 Gradio webui
- **结论：** 需要额外 ~50-100MB 显存，短文本场景收益不确定

### AR 层数压缩（24 → 16/12）

- **16 层 CE 训练 100ep：** acc=0.648，漏段依旧；10ep 与 100ep 效果相同 → 容量上限问题，非训练量问题
- **结论：** 24 层教师模型不可缩减，容量是瓶颈

### 知识蒸馏（KD）

- **12 层学生 130ep：** acc=0.907，但音质极差
- **16 层学生 20ep：** 音质同样极差
- **结论：** KD 对 AR 模型不可行，高 acc 不代表可用音质

### 12 层蒸馏学生延迟

- **首包 15 字 116ms**（教师 220ms，-47%）
- 延迟收益显著，但音质不可接受

## 代码变更记录

| 文件 | 修改内容 |
|------|----------|
| `pipeline/tts.py` | soundfile 替代 torchaudio.load；去掉 super_sampling/vocoder；audio_postprocess 去参数；去掉 @torch.jit.script |
| `pipeline/text_preprocessor.py` | 仅保留中文分支 |
| `text/chinese2.py` | G2PW 延迟加载 |
| `text/cleaner.py` | 仅保留中文 + v2 分支 |
| `text/LangSegmenter/langsegmenter.py` | 移除自定义 fast_langdetect 缓存路径 |
| `server.py` | TTS_Config 嵌套结构；env vars 在 import 前设置 |
| `models/t2s_lightning_module.py` | 去掉训练方法 |
| `models/t2s_utils.py` | 去掉 DPO 相关函数 |
| `models/vits.py` | 去掉 V3/CFM/Discriminator |
| `i18n/i18n.py` | 简化为 passthrough stub |
