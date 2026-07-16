# CHANGES.md — thin-tts-server 优化与魔改记录

本文档记录了从 GPT-SoVITS 原始代码到 thin-tts-server 独立包过程中的所有修改，包括裁剪、Bug 修复、性能优化，以及已尝试但不可行的方向。

## v1.1.0 — 2026-07-16

- 新增 Textual TUI，集中提供配置编辑、启动阶段监控、请求配置与实际后端对照、日志、只读诊断和显式启停操作。
- TUI 只作为启动器：服务通过独立会话和持久状态文件运行，关闭 TUI 不会停止服务；重新打开后可继续监控，只有显式停止才终止进程。
- 新增 `serve | tui | doctor | status | stop` 子命令，同时保留旧 `thin-tts-server --config ...` 调用方式。
- 新增 `max-performance | balanced | compatible` 预设和 `fail | warn | allow` 回退策略。最大性能预设强制 Triton、G2PW CUDA、FP16、RNG 隔离和 VITS 文本缓存。
- 启动过程输出版本化 JSON 阶段事件；失败主界面只显示原因和可能解决办法，原始堆栈保留在技术日志中。诊断和 TUI 均不会安装或修复依赖。
- `/health` 新增生效配置报告；补充 `torchmetrics`，限制 `transformers<5`，并锁定与现有依赖兼容的 Textual/Rich 范围。

## v1.0.0 — 2026-07-15

- 将经过完整质量验证的 buffer-aware Mode 4 与请求级 RNG 隔离设为 `/stream` 默认合同；API 只接受模式 2/3/4，并拒绝未知字段，避免请求级参考音频等旧参数被静默忽略。
- Mode 4 默认改为 500 ms 播放缓冲水位调度：首块及低水位时保留 50-token 上限，缓冲充足后等待自然静音边界；deadline 在每个 semantic chunk 内冻结，避免 GPU 墙钟抖动改变切块。显式 `50/50`、`50/120` 仍可回退。long 从 16 块降至稳定 3 块、强制接缝从 3 降至 0，总耗时中位数 `1870.0 → 1648.0 ms`，最小预测缓冲余量 497.9 ms；用户试听确认后半段更稳定、失真更少且更干净。
- 纯中文输入绕过语言检测；RoBERTa 改用 base model，跳过未使用的 MLM head，前端输出保持一致。
- T2S/VITS 使用相互独立的请求级随机数生成器，同文本、同 seed 在不同切块策略和请求历史下保持 semantic 一致。
- PCM16 转换在缩放前裁剪到 `[-1, 1]` 并乘 32767，消除正满幅环绕到 -32768 的单样本爆音。
- 新增显式 fixed-voice profile：首次 warmup 原子编译约 1 MiB 的 prompt semantic/reference/SV/prompt frontend cache，后续启动在模型初始化前验证 schema、模型/参考音频/参考文本 fingerprint，命中时跳过 CN-HuBERT 与 ERes2Net。实测释放约 252 MiB 可用显存，卸载前后 PCM bitwise equal；状态通过 `/health` 暴露。
- VITS 流式链路默认缓存固定 phones 的 `text_embedding + encoder_text` 输出，并保留请求级回退。long VITS 累计中位数改善约 20.3%，端到端改善约 5.7%，PCM bitwise equal。
- 修复流式合成最后一个音频 chunk 未经过 SOLA crossfade，导致尾部拼接出现爆音的问题。final chunk 使用完整 overlap 做 SOLA 对齐，但仅用 1 ms Hann 前沿完成实际混合，避免把上一块尾部的异常波形继续带入。opening 边界单点跳变从 `19592` 降至 `1376`（-93.0%），边界后 2 ms 最大跳变从 `10811` 降至 `4424`，未增加首包延迟。
- 将 G2PW CUDA Provider 接入正式链路，支持 `auto | cpu | cuda`、可选显存上限、CUDA 初始化失败自动回退 CPU，并通过 `/health` 与结构化日志暴露实际 Provider。固定已验证的 `onnxruntime-gpu==1.23.2`，移除会传递安装 CPU ORT 的冗余外部 `g2pw` 依赖；CPU/CUDA 注音输出一致，热态 G2PW 约快 7 倍，Full Graph 首包实测从 `158.5/195.1 ms` 降至 `111.0/130.4 ms`，额外显存约 1.2 GB。
- 新增 MIT 主许可证、Apache-2.0 全文和第三方来源声明；源码、wheel 元数据与健康检查统一报告 1.0.0。

## v0.2.0 — 正式 Linux/Triton 后端

- 将动态 `seq_len` Triton attention 和 24 层 Full CUDA Graph 提升为正式可选后端。
- `auto | sdpa | triton` 配置支持 CLI、YAML 与环境变量；Windows/CPU/FP32/Triton 缺失时自动使用 SDPA。
- prompt 直接复用固定地址 KV cache，warmup 捕获一次后跨请求、跨长度复用同一张 Graph。
- 捕获、输入形状或 replay 失败时恢复 KV 长度并重算当前 token，随后保持 SDPA fallback。
- `/health` 和结构化日志暴露 active backend、capture count、Graph 显存与 fallback reason。
- 独立包可加载旧 SoVITS checkpoint 中的 `utils.HParams`，无需把完整 GPT-SoVITS 加入 `sys.path`。
- 修复未显式传入 CLI 参数时，默认 host、port、device、half 覆盖 YAML 配置的问题。
- 正式路径 5 次实测：opening 首包 `198.6 → 146.7 ms`（-26.1%），long stress `265.5 → 193.3 ms`（-27.2%）；完整生成下降 52–54%。

## v0.1.1 — 推理稳定性与 Linux/Triton POC

**生产路径（API 与采样默认值不变）：**

- 共享 pipeline 在完整流生命周期内串行化，隔离 KV cache、随机数与参考音频状态。
- 移除每请求 `gc.collect()` / `torch.cuda.empty_cache()`，保留显式 OOM/模型切换清理入口。
- BERT phone-level 特征保持在 GPU，并使用同设备索引展开。
- KV cache 改用 `torch.empty`，避免清零不会读取的 1536-token 尾部。
- 服务默认关闭逐 token tqdm 输出，减少同步打印开销。

**实验性 Linux/Triton POC（尚未作为生产默认后端）：**

- 单张 CUDA Graph 通过设备端 `seq_len` 标量覆盖动态 KV 长度，并融合当前 token 的 K/V 写入。
- RTX 4080 Laptop / WSL2 实测 Mode 4 首包：opening `216.6 → 154.8 ms`（-28.5%），long stress `271.3 → 187.7 ms`（-30.8%）。
- 完整生成中位数下降约 54–55%；Graph 仅捕获一次，额外显存约 8.13 MB。
- 6 组盲听质量测试未发现 Triton 引入的音质退化；共同出现的一处尾部爆音同时存在于 SDPA 与 Triton。

完整 POC 说明见 `prototypes/linux_cuda_graph_backend_poc/NOTES.md`。

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
