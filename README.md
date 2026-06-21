# thin-tts-server

基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2ProPlus 的**精简流式 TTS 推理服务器**。仅保留推理代码、v2ProPlus 版本、中文支持，去掉训练代码、多版本分支、多语言 G2P、BigVGAN vocoder 等冗余模块。

打包为独立 Python wheel（~1.1MB），安装后即可通过 HTTP API 进行流式语音合成。

## 环境要求

- Python 3.10+
- NVIDIA GPU（需 CUDA 支持）
- CUDA Toolkit 11.8+ / 12.x
- PyTorch 2.1+（需匹配 CUDA 版本）
- 显存建议 ≥ 4GB（模型运行时约占 2-3GB）

## 快速开始

### 1. 安装 PyTorch（CUDA 版）

如果还没装 CUDA 版 PyTorch，先去 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选对应 CUDA 版本的命令，例如：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装 thin-tts-server

```bash
pip install dist/thin_tts_server-0.1.0-py3-none-any.whl
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
# 服务器配置
host: "0.0.0.0"
port: 9881
device: "cuda"
half: true                # 使用 FP16 推理（推荐，节省显存）

# 模型权重路径（改为你的实际路径）
t2s_weights: "D:/GPT-SoVITS/GPT_weights_v2ProPlus/your_model-e14.ckpt"
vits_weights: "D:/GPT-SoVITS/SoVITS_weights_v2ProPlus/your_model_e10_s1290.pth"
# vits_lora: ""           # 如有 LoRA 权重，取消注释并填入路径
bert_path: "D:/GPT-SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
hubert_path: "D:/GPT-SoVITS/pretrained_models/chinese-hubert-base"
sv_path: "D:/GPT-SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
```

### 5. 启动服务

```bash
thin-tts-server --config config.yaml
```

也可以用命令行参数覆盖配置：

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
# {"status":"ok","loaded":true,"streaming":true}
```

**流式合成：**

```bash
curl -X POST http://localhost:9881/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是 thin-tts-server 的测试。",
    "ref_audio_path": "D:/reference_audio.wav",
    "ref_text": "这是参考音频对应的文本内容。",
    "mode": 3,
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
{"status": "ok", "loaded": true, "streaming": true}
```

### POST /stream

流式语音合成。返回 `audio/wav` 流，客户端可边接收边播放。

**请求体（JSON）：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | — | 待合成文本 |
| `ref_audio_path` | string | 是 | — | 参考音频路径（.wav） |
| `ref_text` | string | 是 | — | 参考音频对应文本 |
| `mode` | int | 否 | 3 | 文本切分模式（0-5），推荐 3 |
| `seed` | int | 否 | 8110 | 随机种子，影响生成多样性 |
| `speed_factor` | float | 否 | 1.0 | 语速倍率 |
| `top_k` | int | 否 | 15 | AR 采样 top_k |
| `top_p` | float | 否 | 1.0 | AR 采样 top_p |
| `temperature` | float | 否 | 1.0 | AR 采样温度 |
| `fragment_interval` | float | 否 | 0.3 | 分句间隔（秒） |

**响应：** `Content-Type: audio/wav`，流式返回 PCM16 WAV 数据。

### POST /tts

非流式合成（一次性返回完整音频）。参数同 `/stream`。

## 配置优先级

配置值的优先级从高到低：

1. **命令行参数**（`--port 9882`）
2. **环境变量**（`THIN_TTS_PORT=9882`）
3. **YAML 配置文件**（`port: 9882`）
4. **默认值**

支持的环境变量：`THIN_TTS_HOST`、`THIN_TTS_PORT`、`THIN_TTS_DEVICE`、`THIN_TTS_HALF`、`THIN_TTS_T2S_WEIGHTS`、`THIN_TTS_VITS_WEIGHTS`、`THIN_TTS_BERT_PATH`、`THIN_TTS_HUBERT_PATH`、`THIN_TTS_SV_PATH`。

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

本项目基于 GPT-SoVITS，遵循其原始许可证。
