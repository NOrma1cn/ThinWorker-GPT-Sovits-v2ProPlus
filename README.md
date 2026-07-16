# thin-tts-server

基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2ProPlus 的**精简流式 TTS 推理服务器**。仅保留推理代码、v2ProPlus 版本、中文支持，去掉训练代码、多版本分支、多语言 G2P、BigVGAN vocoder 等冗余模块。

打包为独立 Python wheel，安装后即可通过 HTTP API 进行流式语音合成。1.1 默认提供最大性能预设、请求级 T2S/VITS 随机数隔离、VITS 文本编码缓存，以及可配置的降级策略。

## 环境要求

- Windows 11 主机；正式 Triton 链路运行在 WSL2 Linux
- NVIDIA RTX 40 系列 GPU，以及支持 WSL CUDA 的 NVIDIA 驱动
- WSL 内 Python 3.10+
- CUDA 版 PyTorch 2.1+（版本需与驱动兼容）
- 显存建议 ≥ 4 GB；G2PW CUDA 还会额外占用约 1.2 GB

原生 Windows Python 可以使用 SDPA 回退链路，但无法启用正式 Triton Full Graph。分发给其他机器时应将 WSL2、驱动和 CUDA 版 PyTorch 作为部署前置条件，而不是把本机虚拟环境整体复制过去。

## T2S 推理后端

`t2s_backend` 支持三种模式：

- `auto`（默认）：Linux + CUDA + FP16 + Triton 可用时启用 Full CUDA Graph，否则自动回退 SDPA。
- `sdpa`：始终使用 PyTorch SDPA，适合 Windows 或故障排查。
- `triton`：优先使用 Linux/Triton；不兼容或运行失败时仍回退 SDPA。

可以在 YAML 的 `server` 段配置：

```yaml
server:
  device: cuda
  half: true
  t2s_backend: auto
```

也可以使用 `--t2s-backend auto|sdpa|triton` 或环境变量
`THIN_TTS_T2S_BACKEND`。调试时设置 `THIN_TTS_T2S_BACKEND_STRICT=1`，
会在强制 Triton 不可用或运行失败时直接报错，而不是回退。

服务 warmup 会完成首次 Graph 捕获。`/health` 的 `t2s_backend` 字段会报告
当前后端、捕获次数、Graph 显存及回退原因。当前 Triton 路径针对 batch=1、
FP16、单 token AR decode；遇到不兼容形状会自动切回 SDPA。

## G2PW 推理后端

`g2pw_backend` 控制中文多音字模型使用的 ONNX Runtime Provider：

- `auto`（默认）：优先 CUDA，CUDA Provider 缺失或初始化失败时自动回退 CPU。
- `cpu`：始终使用 CPU，适合显存紧张或排查 CUDA 依赖问题。
- `cuda`：优先 CUDA，并保留 CPU 自动回退以保证服务可用。

项目固定使用已验证的 `onnxruntime-gpu==1.23.2`。该发行包同时包含 CPU
Provider，因此不需要并装 `onnxruntime`；从旧环境升级时应先卸载 CPU 包，
再安装当前项目，避免两个发行包覆盖同一个 Python 模块：

```bash
pip uninstall -y onnxruntime
pip install .
```

可以在 YAML 的 `server` 段配置：

```yaml
server:
  g2pw_backend: auto
  # 可选；不设置时由 ONNX Runtime 管理显存
  g2pw_cuda_memory_limit_mb: 1536
```

对应命令行参数为 `--g2pw-backend auto|cpu|cuda` 和
`--g2pw-cuda-memory-limit-mb`；环境变量为 `THIN_TTS_G2PW_BACKEND` 和
`THIN_TTS_G2PW_CUDA_MEMORY_LIMIT_MB`。CUDA G2PW 在 RTX 4080 Laptop 实测
额外占用约 1.2 GB 显存，热态速度约为 CPU 的 7 倍。`/health` 的
`g2pw_backend` 字段会报告请求模式、实际 Provider 和回退原因。

## 快速开始

### 1. 安装 PyTorch（CUDA 版）

如果还没装 CUDA 版 PyTorch，先去 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选对应 CUDA 版本的命令，例如：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装 thin-tts-server

```bash
pip install dist/thin_tts_server-1.1.0-py3-none-any.whl
```

或从源码安装：

```bash
pip install .
```

### 3. 准备模型权重

你需要准备以下 5 组权重/模型文件：

| 名称 | 说明 | 来源 |
|------|------|------|
| **T2S 权重** | AR 模型（Text-to-Semantic），`.ckpt` 文件 | GPT-SoVITS 训练产出，或社区分享 |
| **VITS 权重** | VITS 解码器，`.pth` 文件（支持 LoRA 权重） | GPT-SoVITS 训练产出，或社区分享 |
| **BERT 模型** | chinese-roberta-wwm-ext-large 目录 | [HuggingFace](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) |
| **HuBERT 模型** | chinese-hubert-base 目录 | [HuggingFace](https://huggingface.co/TencentGameMate/chinese-hubert-base) |
| **SV 模型** | Speaker Verification，`.ckpt` 文件 | GPT-SoVITS `pretrained_models/sv/` 目录 |

此外，你还需要一段**参考音频**（`.wav`）和对应的**参考文本**来指定目标音色。

> 首次启动时会自动下载 G2PW ONNX 模型（~635MB）和 fast_langdetect 模型（~125MB），请耐心等待。

### 4. 创建配置文件

创建 `config.yaml`：

```yaml
server:
  preset: "max-performance"
  host: "0.0.0.0"
  port: 9881
  device: "cuda"
  half: true                # 使用 FP16 推理（推荐，节省显存）
  t2s_backend: "triton"
  g2pw_backend: "cuda"
  g2pw_cuda_memory_limit_mb: 1536
  fallback_policy: "fail"
  rng_isolation: true
  cache_vits_encoded_text: true

weights:
  # WSL 路径；改为你的实际位置
  t2s_weights: "/mnt/d/GPT-SoVITS/GPT_weights_v2ProPlus/your_model-e14.ckpt"
  vits_weights: "/mnt/d/GPT-SoVITS/SoVITS_weights_v2ProPlus/your_model_e10_s1290.pth"
  # vits_lora: ""           # 如有 LoRA 权重，取消注释并填入路径
  bert_path: "/mnt/d/GPT-SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
  hubert_path: "/mnt/d/GPT-SoVITS/pretrained_models/chinese-hubert-base"
  sv_path: "/mnt/d/GPT-SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
  ref_audio: "/mnt/d/thin-tts/reference.wav"
  ref_text: "参考音频对应文本。"
  # 可选：首次启动自动生成，后续跳过 HuBERT/SV 加载。
  voice_profile: "/mnt/d/thin-tts/voice-profile.pt"
```

`voice_profile` 适用于服务端固定参考音频的部署。文件不存在、模型/参考音频/参考文本发生变化或 profile 不可读时，服务会用完整链路 warmup 后原子重建；命中时直接恢复 prompt semantic、reference spectrogram、speaker embedding 和 prompt frontend cache，并跳过 CN-HuBERT 与 speaker encoder。`/health` 的 `voice_profile` 字段会报告 `disabled`、`compiled` 或 `loaded`。

### 5. 使用 TUI 配置并启动

```bash
thin-tts-server tui --config config.yaml
```

在交互式终端直接运行 `thin-tts-server` 也会打开 TUI。面板可以保存配置、启动/停止/重启服务，并显示启动阶段、请求配置与实际生效状态、技术日志和只读环境诊断。

TUI 只是启动器。服务以独立后台进程运行，关闭 TUI 或终端不会停止服务；只有面板中的“停止”操作或以下命令会停止它：

```bash
thin-tts-server status
thin-tts-server stop
```

诊断只报告原因和可能的解决办法，不会执行 `pip`、`apt`、驱动、CUDA 或其他依赖安装。也可以在终端运行：

```bash
thin-tts-server doctor --config config.yaml
```

无 TUI 的服务器或进程管理器应使用 `serve` 子命令：

```bash
thin-tts-server serve --config config.yaml
```

旧版直接传参数的调用方式仍然兼容。也可以用命令行参数覆盖配置：

```bash
thin-tts-server --config config.yaml --port 9882 --device cuda:1
```

启动后会看到类似日志：

```
Loading Text2Semantic weights from ...
Loading VITS weights from ...
Loading BERT weights from ...
Loading CNHuBERT weights from ...
INFO:     Uvicorn running on http://0.0.0.0:9881
```

模型加载大约需要 1-3 分钟（取决于磁盘速度）。

### 6. 验证

**健康检查：**

```bash
curl http://localhost:9881/health
# {"status":"ok","server":"thin-tts-server","version":"1.1.0","loaded":true,...}
```

**流式合成：**

```bash
curl -X POST http://localhost:9881/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是 thin-tts-server 的测试。",
    "seed": 8110
  }' \
  --output test.wav
```

返回的是 PCM16 WAV 流（采样率由模型决定，通常 32kHz），可直接用音频播放器打开。

## API 文档

### GET /health

返回服务器状态。

**响应示例：**
```json
{"status":"ok","server":"thin-tts-server","version":"1.1.0","loaded":true,"streaming":true,"t2s_backend":{...},"g2pw_backend":{...},"configuration":{...}}
```

### POST /stream

流式语音合成。返回 `audio/wav` 流，客户端可边接收边播放。

**请求体（JSON）：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | — | 待合成文本 |
| `mode` | int | 否 | 4 | `2` 自然静音边界；`3` 固定短块；`4` 缓冲水位混合策略（推荐） |
| `seed` | int | 否 | 8110 | 随机种子，影响生成多样性 |
| `min_chunk_length` | int | 否 | 10 | 静音边界搜索的最小 semantic token 数 |
| `hybrid_switch_tokens` | int | 否 | 50 | Mode 4 首块及低缓冲时的 semantic token 上限 |
| `hybrid_buffer_target_ms` | int | 否 | 500 | Mode 4 预测播放缓冲水位；`0` 关闭动态水位 |
| `hybrid_steady_tokens` | int | 否 | — | 显式固定后续上限；提供该字段时保留旧的固定 deadline 策略 |
| `rng_isolation` | bool | 否 | true | 使用请求级 T2S/VITS RNG，保证同 seed 不受其他请求影响 |
| `cache_vits_encoded_text` | bool | 否 | true | 缓存固定目标文本的 VITS 编码结果 |

**响应：** `Content-Type: audio/wav`，流式返回 PCM16 WAV 数据。

> 服务器使用一个共享的 GPU 推理 pipeline。并发请求会在服务内排队并串行执行，
> 以隔离 T2S KV cache、随机数状态和参考音频缓存，避免请求之间互相污染。

## 配置优先级

配置值的优先级从高到低：

1. **命令行参数**（`--port 9882`）
2. **环境变量**（`THIN_TTS_PORT=9882`）
3. **YAML 配置文件**（`port: 9882`）
4. **默认值**

支持的环境变量包括 `THIN_TTS_PRESET`、`THIN_TTS_HOST`、`THIN_TTS_PORT`、`THIN_TTS_DEVICE`、`THIN_TTS_HALF`、`THIN_TTS_FALLBACK_POLICY`、`THIN_TTS_RNG_ISOLATION`、`THIN_TTS_CACHE_VITS_ENCODED_TEXT`，以及各模型路径对应的 `THIN_TTS_*` 变量。

## 性能预设与降级策略

- `max-performance`：请求 Triton、G2PW CUDA、FP16、RNG 隔离和 VITS 文本编码缓存；任何请求的后端未生效时启动失败。
- `balanced`：保留 Triton 和缓存，将 G2PW 放在 CPU；发生后端降级时明确告警。
- `compatible`：使用 SDPA 与 G2PW CPU，适合有 CUDA 但不支持 Triton 的环境。

新建配置会同时预填音色 Profile 路径，首次 warmup 编译后即可在后续启动跳过音色编码器；旧配置只有明确设置 `weights.voice_profile` 才会启用。预设只提供缺省值，YAML 中的显式字段、环境变量和命令行参数仍可覆盖它。

`fallback_policy` 可独立设置为 `fail`、`warn` 或 `allow`。`/health` 会分别报告请求配置和实际生效的 T2S、G2PW、音色 Profile 与缓存设置，避免静默回退。

## 从源码构建 wheel

```bash
pip install build
python -m build
# 产出在 dist/ 目录
```

## 常见问题

### Q: 首次启动很慢？

正常。首次启动会自动下载两个模型：G2PW ONNX 模型（~635MB）和 fast_langdetect 语言检测模型（~125MB）。下载完成后后续启动不再重复。

### Q: Windows 上报 torchaudio 相关错误？

thin-tts-server 已针对 Windows 做了适配——用 `soundfile` 库替代了 `torchaudio.load`。如果遇到其他 torchaudio 相关问题，确保安装了 `soundfile`：

```bash
pip install soundfile
```

### Q: 显存不够怎么办？

- 确保 `half: true`（FP16 推理）已开启
- 减少单次请求的文本长度
- 关闭其他 GPU 占用程序

### Q: 参考音频有什么要求？

- 格式：WAV（PCM16 或 float32）
- 时长：建议 3-10 秒
- 质量：干净、无噪音、无背景音乐
- 采样率：不限（内部会重采样）

### Q: 支持英文或其他语言吗？

当前版本仅支持中文。如需多语言支持，请基于完整 GPT-SoVITS 项目。

## 许可证

项目自身修改采用 [MIT License](LICENSE)。上游来源、Apache-2.0 与其他 MIT 组件的归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)。模型权重与自动下载的模型资产可能有独立许可，分发前需另行核对。
