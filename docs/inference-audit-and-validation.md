# Thin TTS v2ProPlus 推理全链路审查与验证路线

审查日期：2026-07-14  
审查分支：`release/triton-backend`  
审查范围：`thin-tts-server-triton-release` 的中文、固定音色、流式 GPT-SoVITS v2ProPlus 正式链路。

## 1. 结论

项目不应立即停止。当前链路的热态首包已经很接近这一代 GPT-SoVITS 架构在 RTX 40 系显卡上的实用上限，但仍存在：

1. 一个能够确定复现的 PCM16 正峰值溢出问题，可能制造单样本爆音。
2. 一个中文-only 路径中完全无效的语言识别阶段。
3. 一个只取 hidden state 却仍计算 MLM logits 的 RoBERTa 冗余输出头。
4. 多个可以严格缓存的固定音色与 VITS 静态条件。
5. 一个随长文本累积的 VITS 语义前缀重复编码问题。
6. 一条值得同机盲测的模型换代路线，但没有公开结果能够直接证明它会击败当前 111 ms 的 opening 首包。

短期无训练优化预计还能降低约 5-15 ms 首包，并降低约 15-25% 长文本总耗时。要进一步获得数量级变化，需要重训 causal/chunk-aware 声学解码器，或者迁移到新的模型家族。

## 2. 当前端到端链路

```text
HTTP POST /stream
  -> singleton pipeline lock
  -> fixed reference/prompt cache
  -> text split and Chinese text normalization
  -> split-lang / fast_langdetect
  -> G2PW polyphone prediction
  -> Jieba POS, pronunciation overrides, tone sandhi, erhua
  -> OpenCPOP phone IDs
  -> Chinese RoBERTa-large phone-level features
  -> T2S autoregressive semantic generation
  -> Mode 4 mute-boundary/deadline scheduler
  -> SoVITS v2ProPlus streaming decode
  -> latent overlap and SOLA waveform splice
  -> PCM16 WAV stream
```

固定参考音频有三条条件分支：

```text
Reference WAV
  -> CN-HuBERT -> SSL features -> VITS RVQ -> prompt semantic tokens -> T2S
  -> STFT spectrum -> VITS reference encoder -> GE                 -> SoVITS
  -> 80-bin fbank -> ERes2NetV2 -> speaker embedding               -> SoVITS
```

参考文本另走一次与目标文本相同的中文前端，其 phones、BERT feature 和 normalized text 已缓存。

## 3. 模型与职责

| 模型 | 规模 | 在线职责 | 审查结论 |
|---|---:|---|---|
| T2S AR Transformer | 77.6M / 148 MB | phones、BERT、prompt semantic 到 25 Hz semantic token | 保留 Full CUDA Graph；已经很快 |
| SoVITS v2ProPlus | 86.3M / 165 MB | semantic、phones、GE、SV 到 32 kHz waveform | 当前最值得继续优化的阶段 |
| Chinese RoBERTa-large | 约 325M / 621 MB | 1024 维上下文文本特征 | 暂留；跳过 MLM head；长期蒸馏或共享编码器 |
| CN-HuBERT base | 约 95M / 180 MB | 固定参考音频到 prompt semantic | 编译 voice profile 后可卸载 |
| ERes2NetV2 | 53.6M / 103 MB | 固定参考音频到 speaker embedding | 编译 voice profile 后可卸载 |
| G2PW ONNX | 606 MB | 中文多音字消歧 | 保留 CUDA Provider；长期可与 RoBERTa 共享编码器 |
| fast_langdetect | 首次下载约 125 MB | 语言检测 | 中文-only 路径中删除 |

SoVITS 当前配置为 25 Hz、1024-entry 单层 RVQ、32 kHz 输出，upsample rates 为 `[10, 8, 2, 2, 2]`。推理仍使用 `noise_scale=0.5` 的 Gaussian latent noise。

## 4. 已有优化与实测基线

已进入正式链路：

- Linux Triton dynamic attention 和 24 层 Full CUDA Graph。
- 静态持久 KV cache 与 device-side sequence length。
- SDPA 自动回退。
- G2PW CUDA Provider 与 CPU 回退。
- RoBERTa phone feature 向量化扩展。
- prompt/reference/SV cache。
- request-local GE cache。
- Mode 4 mute-boundary/deadline 调度。
- 最后一个 chunk 的 SOLA 修复与 1 ms fade。
- 整个 request 生命周期持有 pipeline lock。

G2PW CUDA 后的热态基线：

| 指标 | Opening | Long |
|---|---:|---:|
| TTFB | 约 111.0 ms | 约 130.4 ms |
| T2S first yield | 约 37-43 ms | 约 37-43 ms |
| First VITS | 约 30-32 ms | 约 30-32 ms |
| RoBERTa | 约 10 ms | 约 10 ms |
| G2PW CUDA | 约 3.84 ms/句 | 约 3.84 ms/句 |

G2PW CPU 到 CUDA 为 `26.86 -> 3.84 ms/句`，约 7 倍；代价约为 1.2 GB 额外 GPU memory。Full Graph 相对 SDPA 的 opening/long TTFB 改善约 26-31%，完整生成改善约 52-55%。

## 5. 已证实问题

### 5.1 PCM16 正峰值溢出

`audio_postprocess` 会把超出范围的 chunk 除以绝对峰值，因此正峰值可能精确等于 `+1.0`。随后执行：

```python
(audio * 32768).astype(np.int16)
```

验证结果：

```text
input:   [0.99999, 1.0, 1.00001, -1.0]
current: [32767, -32768, -32768, -32768]
safe:    [32766, 32767, 32767, -32767]
```

这会把正满幅样本翻转成负满幅样本，能够制造单样本的大幅跳变。正确策略是先 clamp 到 `[-1, 1]`，再乘 `32767`。

### 5.2 中文-only 语言识别无效

`TextPreprocessor` 调用 `LangSegmenter.getTexts(text, "zh")`。`LangSegmenter` 仍先运行 `split_by_lang`，但因为设置了 `default_lang="zh"`，所有识别结果之后都会被覆盖为 `zh`。

因此 fast_langdetect 不影响最终 language tag，只增加启动下载、依赖和热路径工作。删除时需要用 golden corpus 验证中文、数字、标点、ASCII 混输的 normalized text、phones 与 BERT feature 完全一致。

### 5.3 RoBERTa MLM 输出头无消费者

当前加载 `AutoModelForMaskedLM`，forward 后只读取倒数第三层 hidden state。Masked-LM prediction head 的 vocabulary logits 没有消费者。

优先验证直接调用现有模型的 `base_model`，以避免 checkpoint 加载差异。成功门槛是 hidden state bitwise equal 或在当前 dtype 下数值完全等价，随后再测延迟。

## 6. 结构性性能问题

### 6.1 VITS 累计语义前缀重复编码

流式循环保存 `previous_tokens`，每个 chunk 都重新 `torch.cat` 成完整累计 semantic prefix。`decode_streaming` 随后对完整 prefix 执行：

```text
RVQ decode
-> semantic encoder_ssl
-> text encoder
-> MRTE
-> encoder2
-> crop result_length
-> latent sampling
-> flow
-> waveform decoder
```

需要注意：crop 发生在 `enc_p` 尾部，因此 flow 和 waveform decoder 只处理当前 result tail；真正随累计长度重复的是 crop 之前的 semantic/text encoder 路径。

可严格等价缓存：

- 目标 phones 的 text embedding、text mask、encoder_text 输出。
- 固定参考音色的 GE、GE512 和 SV embedding。
- 已解码 RVQ embedding 的历史部分，虽然该部分占比预计较小。

不能直接严格缓存：

- `encoder_ssl`、MRTE 和 `encoder2` 的历史输出，因为它们不是为 causal streaming 训练，新增 token 可能改变历史 hidden state。

无训练近似方案是只保留最近 3-6 秒 semantic context；必须独立盲听长句、爆破音、语气连续性和边界。

### 6.2 全局 RNG 绑定了 T2S、VITS 与 chunk 策略

每个请求会重置 Python、NumPy、Torch 和 CUDA 全局 RNG。T2S sampling 与 VITS `randn_like` 消耗同一 CUDA RNG，因此改变 chunk 数、并行顺序或 VITS 调用形状会改变后续 semantic token。

仅隔离 T2S/VITS generator 可以固定 semantic token，但仍不能保证不同 chunk 方案产生相同音频。要进一步做到 chunk-schedule-invariant，需要让 VITS noise 由 request seed 与绝对 latent frame position 决定。

### 6.3 固定音色参考模型常驻

服务 API 不接受动态参考音频；reference audio 和 prompt text 来自 server config，并在 warmup 使用。可以预编译 voice profile：

- `prompt_semantic`
- `refer_spec` 或最终 `GE/GE512`
- `sv_embedding`
- prompt phones、BERT feature、normalized text
- T2S/VITS/BERT/HuBERT/SV checkpoint fingerprint
- profile schema version

profile 验证成功后可以卸载 CN-HuBERT 和 ERes2Net。该优化主要降低 VRAM 与启动常驻成本，不应承诺明显热态 TTFB 收益。

## 7. 阶段取舍矩阵

| 阶段 | 决策 | 无训练动作 | 需要训练的替代 |
|---|---|---|---|
| 中文 TN | 保留 | 扩充领域 golden corpus | WeTextProcessing/NeMo WFST |
| fast_langdetect | 删除 | 中文-only 直接处理全文 | 不适用 |
| G2PW | 保留 | CUDA、confidence 暴露、领域词典 | 与文本 encoder 共享 backbone |
| RoBERTa-large | 优化 | 跳过 MLM head | 蒸馏或 shared dual-head encoder |
| CN-HuBERT/ERes2Net | 离线化 | voice profile 后卸载 | 新 speaker/style encoder |
| T2S AR | 保留 | Triton Full Graph、静态 generator | 12 层蒸馏、multi-token heads |
| SoVITS | 重点优化 | 静态条件 cache、shape bucket、bounded context | causal/chunk-aware decoder |
| SOLA/PCM | 修复并保留 | safe PCM16、boundary click detector | 不适用 |
| Pipeline lock | 暂留 | 独立 session state 后再拆 | worker pool / per-session runtime |

## 8. 外部替代方案

### 8.1 最相关：Fun-CosyVoice 3 / CosyVoice 2

CosyVoice 2 从训练阶段采用 chunk-aware causal flow，支持 streaming 与 non-streaming。当前 Fun-CosyVoice 3 README 还提供中文拼音 inpainting、text-in/audio-out 双流式，并声称最低约 150 ms latency。

它是当前最值得同机比较的完整模型家族，但 150 ms 的公开数字并不直接优于本项目 111 ms opening。迁移动机应是长文本稳定性、韵律、发音控制和维护成本，而不是先假设首包更快。

### 8.2 ZipVoice

ZipVoice 使用 compact Zipformer 和 flow distillation，支持中文/英文。论文报告相对 DiT flow baseline 最多快 30 倍，官方 TensorRT 路径声称 GPU throughput 约 2 倍。当前公开材料更适合比较完整生成速度，不能把 throughput 或 RTF 当作 TTFB。

### 8.3 Fish Speech、F5-TTS、IndexTTS2

- Fish Speech 公布 RTF 0.195、较低中文 WER 与 streaming 路径，但模型使用自定义研究许可证。
- F5-TTS 论文报告 RTF 0.15，但原始模型不是首包优先的真正流式设计，公开权重为 CC-BY-NC。
- IndexTTS2 的价值是情绪与 duration 解耦，适合数字人同步，但它仍是 AR 模型，不应作为纯延迟替代。

### 8.4 Codec 与 vocoder

- Mimi 以 12.5 Hz 运行，理论上能把当前 25 Hz semantic AR step 减半，但 codec frame 自身为 80 ms，而且替换它需要重训 T2S 和声学 decoder。
- SpeechTokenizer、EnCodec、DAC 同样不是当前 RVQ/SoVITS 的即插即用替代。
- BigVGAN 和 Vocos 接收 mel 或各自 codec feature，不能直接消费当前 SoVITS latent/GE，需要重训或蒸馏。

## 9. 按顺序验证

### V1 PCM16 correctness

1. 添加 `+1.0`、`-1.0`、超范围正负样本测试。
2. 先观察测试在当前实现失败。
3. 改为 clamp 与 `32767` conversion。
4. 跑完整测试。
5. 生成原版/修复版 opening、long 样本；统计 clipping、最大单点跳变并试听。

成功门槛：不再发生正峰值 wraparound；没有新增失真；现有 SOLA 测试通过。

### V2 Chinese-only frontend parity

1. 建立 golden corpus：数字、日期、金额、量词、多音字、儿化、标点、ASCII 混输。
2. 记录 normalized text、pinyin、phones、word2ph。
3. 绕过 LangSegmenter 后逐项比对。
4. 移除 fast-langdetect/split-lang/cn2an，并测冷启动、包体和热态 frontend。

成功门槛：golden 输出完全一致；首次下载不再包含语言模型；热态无退化。

### V3 RoBERTa base-model path

1. 同一模型实例分别运行 MLM wrapper 和 `base_model`。
2. 比较所取 hidden layer。
3. 比较 phones 展开后的最终 BERT feature。
4. 测 opening/long latency 和 peak memory。

成功门槛：feature 一致；至少消除无用 logits，且没有性能回退。

### V4 RNG isolation and adaptive scheduler

1. T2S 与 VITS 使用独立 generator。
2. 固定 T2S semantic token hash。
3. 对比 `50/50` 与 `first=50/steady=120`。
4. 再评估按绝对 latent position 生成 VITS noise。

成功门槛：semantic token 不随 chunk 策略变化；opening TTFB 不退化超过 5 ms；long total 至少改善 10%；盲听无新增边界、叠音或语气漂移。

### V5 VITS static conditioning cache

分别缓存 GE512 和 encoded target text，每次只引入一个变量。

成功门槛：PCM bitwise equal；long total 有稳定改善。若改善落入噪声，拒绝该优化。

### V6 Static-shape VITS graph/buckets

对 first chunk 和常用 steady chunk 建立少量 shape buckets，保留 eager fallback。

成功门槛：first VITS 至少改善 15%，无 graph recapture storm，输出通过数值与盲听门槛。

### V7 Bounded semantic context

测试 3、4、6 秒左上下文与完整前缀，重点覆盖长文本和用户已经发现过的咬字、爆破音、opening 漂移案例。

成功门槛：long total 至少改善 20%；用户盲听无可辨质量下降。否则只接受重训 causal decoder，不把窗口近似加入正式链路。

### V8 Offline voice profile

成功门槛：同 seed semantic token 相同、PCM 相同；checkpoint/reference/prompt 任一变化都会使 profile 失效；常驻显存明确下降。

### V9 Model-family benchmark

优先比较 Fun-CosyVoice 3；如许可允许，再加入 ZipVoice 或 Fish Speech。所有模型统一 RTX 4080 Laptop、文本、参考音频、warmup、五次重复与用户盲听。

必须分别报告：first audio byte、首个可播放 chunk、完整生成时间、RTF、VRAM、CER、speaker similarity、边界异常和主观质量。不得用 throughput 或论文 RTF 替代本项目 TTFB。

## 10. 停止条件

完成 V1-V8 并至少对比一个现代模型家族后，如果：

- 无训练优化的 opening/long TTFB 总收益不足 10 ms；
- long total 收益不足 20%；
- 新模型没有在用户盲听、稳定性和可分发许可上同时胜出；

则冻结当前架构，将项目转入兼容性、质量回归和安全维护，不再继续做高复杂度推理微优化。

## 11. 资料来源

- G2PW: https://arxiv.org/abs/2203.10430
- WeTextProcessing: https://github.com/wenet-e2e/WeTextProcessing
- CosyVoice 2: https://arxiv.org/abs/2412.10117
- Fun-CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- ZipVoice: https://arxiv.org/abs/2506.13053
- Fish Speech: https://github.com/fishaudio/fish-speech
- F5-TTS: https://arxiv.org/abs/2410.06885
- IndexTTS2: https://arxiv.org/abs/2506.21619
- Mimi/Moshi: https://github.com/kyutai-labs/moshi
- SpeechTokenizer: https://arxiv.org/abs/2308.16692
- BigVGAN: https://github.com/NVIDIA/BigVGAN
- Vocos: https://github.com/gemelo-ai/vocos

## 12. 验证记录

### 2026-07-14：V1 PCM16 correctness

状态：代码级验证通过；正式模型试听待下次启动 WSL 推理服务时补充。

红灯测试确认：

- `+1.0` 在旧实现中输出 `-32768`。
- `[2.0, -2.0, 1.0, -1.0]` 经峰值归一化后，正峰值发生极性翻转。

最小修复：

```python
(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
```

绿灯结果：

- 新增 2 个 PCM conversion 测试通过。
- 完整测试为 `32 passed, 1 skipped`。
- SOLA、T2S backend、G2PW backend 和 server serialization 现有测试无回归。
- WSL2 当前处于 stopped 状态，因此本阶段没有为试听而临时启动模型或修改配置。
