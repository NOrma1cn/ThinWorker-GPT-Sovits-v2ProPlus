# Thin TTS v2ProPlus 推理全链路审查与验证路线

审查日期：2026-07-14  
审查分支：`release/triton-backend`  
审查范围：`thin-tts-server-triton-release` 的中文、固定音色、流式 GPT-SoVITS v2ProPlus 正式链路。

## 1. 结论

项目不应立即停止。当前链路的热态首包已经很接近这一代 GPT-SoVITS 架构在 RTX 40 系显卡上的实用上限，但仍存在：

1. 一个能够确定复现的 PCM16 正峰值溢出问题，可能制造单样本爆音。
2. 一个纯中文路径中可严格绕过、但混合 ASCII 时仍会改变 BERT 上下文边界的语言识别阶段。
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
| fast_langdetect | 首次下载约 125 MB | 混合文本分段 | 纯中文输入绕过；混合 ASCII 保留兼容回退 |

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

### 5.2 纯中文语言识别可绕过，混合 ASCII 不能直接删除

`TextPreprocessor` 调用 `LangSegmenter.getTexts(text, "zh")`。纯中文、数字和标点输入最终形成一个 `zh` 段，因此 `split-lang / fast_langdetect` 在这条主路径上没有改变任何下游输入。

但 `full_en` 分支早于 `default_lang="zh"` 覆盖执行，混合 ASCII 会保留英文段边界。虽然 `clean_text_inf(..., "zh")` 和 `get_bert_inf(..., "zh")` 都忽略检测出的 language tag，分段边界仍会让 RoBERTa 分别编码多个中文片段。实测强制整句处理会保持 normalized text、G2PW 拼音、phones 和 word2ph 不变，却改变最终 phone-level BERT feature。

因此正式方案是：对当前 normalizer 明确支持的基本汉字、ASCII 数字、空白、标点以及 `￥/^` 直接返回一个 `zh` 段；ASCII 字母、日文、emoji、扩展汉字等全部按需加载旧分段器回退。当前不能从安装依赖中彻底删除 `fast-langdetect/split-lang`。

### 5.3 RoBERTa MLM 输出头无消费者

当前加载 `AutoModelForMaskedLM`，forward 后只读取倒数第三层 hidden state。Masked-LM prediction head 的 vocabulary logits 没有消费者。

已验证直接调用同一模型实例的 `base_model`：FP16 hidden state 在 28、37、59 token 三种长度上均 bitwise equal，最大绝对误差为 0。正式链路因此跳过 MLM prediction head，checkpoint 加载方式和所取 hidden layer 不变。Windows 独立基准的相对收益约 0-3.5 ms，但绝对耗时受当前 GPU 状态影响较大，最终端到端收益需要在 Linux 正式链路复测。

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
| fast_langdetect | 条件绕过 | 纯中文直接处理全文，混合文本兼容回退 | 不适用 |
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
4. 混合 ASCII 单独验证整句处理是否改变 BERT 上下文。
5. 仅在完整等价时删除依赖；否则保留按需加载的兼容回退。

成功门槛：正式路径 golden 输出完全一致；纯中文首次请求不加载语言模型；热态无退化。

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

### 2026-07-14：V2 Chinese-only frontend parity

状态：保守纯中文快速路径通过代码级和真实模型验证；完整删除语言分段器被否决。

验证环境：Windows Python 3.11、RTX 4080 Laptop、FP16 Chinese RoBERTa-large、G2PW CPU Provider。此环境只用于等价性验证，耗时不能与 Linux Triton 正式链路直接比较。

golden corpus 覆盖：

- 普通中文。
- 日期、整数、小数、百分比和金额。
- `重新量一遍` 等量词与多音字。
- 标点、儿化。
- `AI模型GPT-SoVITS v2ProPlus发布。` 混合 ASCII。

真实模型结果：

| Case | Legacy segments | Active = legacy | Whole text = legacy | Active segmentation |
|---|---:|---:|---:|---:|
| plain | 1 | yes | yes | 0.026 ms |
| date | 1 | yes | yes | 0.013 ms |
| numbers | 1 | yes | yes | 0.027 ms |
| amount | 1 | yes | yes | 0.023 ms |
| quantifier | 1 | yes | yes | 0.012 ms |
| polyphones | 1 | yes | yes | 0.011 ms |
| punctuation | 1 | yes | yes | 0.021 ms |
| erhua | 1 | yes | yes | 0.011 ms |
| mixed_ascii | 6 | yes | **no** | 0.650 ms（legacy fallback） |

每个 `yes` 都同时要求 normalized text、G2PW 拼音、phones、phone IDs、word2ph、BERT shape、BERT tensor 和 SHA-256 完全一致。

混合 ASCII 的旧分段为：

```text
AI | 模型 | GPT-SoVITS v | 2 | ProPlus | 发布。
en   zh     en             zh  en        zh
```

强制整句处理后，两条路径均得到：

```text
normalized: 模型减二秒发布.
pinyin:     mo2 xing2 jian3 er4 miao3 fa1 bu4 .
phones:     m o2 x ing2 j ian3 EE er4 m iao3 f a1 b u4 .
word2ph:    2,2,2,2,2,2,2,1
```

但 phone-level BERT SHA-256 分别为：

```text
legacy: c025551f3319cbe73c38b9daa9708e791aa32272fb55765ef73d73e6954c6677
whole:  b91bac6a4b0c9301baab3591e7c5af32495ff5d1f66c56e66cb173e003b9921a
```

这证明语言标签本身虽无消费者，分段边界仍是有效模型输入，不能全局删除。

正式改动：

- 纯中文输入直接形成单个 `zh` 段。
- `LangSegmenter` 改为回退时按需导入，纯中文启动不再触发 language detector。
- ASCII、日文、emoji、扩展汉字等保留旧行为。
- 新增快速路径和回退测试；完整测试为 `34 passed, 1 skipped`。

首次未缓存的 fastText 下载已单独观察到约 125.2 MB 和约 22.3 秒等待；本次缓存后的首个 legacy split 为 468.335 ms，随后约 0.55-0.86 ms。纯中文快速路径为约 0.011-0.027 ms。主要收益是消除首次下载/加载，而不是承诺数毫秒级热态 TTFB 改善。

### 2026-07-14：V3 RoBERTa base-model path

状态：真实模型等价性通过，正式链路已切换为同一 `AutoModelForMaskedLM` 实例的 `base_model` forward。

验证方法：

1. 只加载一次 production Chinese RoBERTa-large。
2. 同一 tokenizer、同一 FP16 权重、同一 CUDA 输入分别运行 MLM wrapper 和 `base_model`。
3. 两条路径都取 `hidden_states[-3][0, 1:-1]`。
4. 对 opening、medium、long 交替运行 50 次，避免固定先后顺序偏差。

结果：

| Case | Tokens | Hidden bitwise equal | Max abs diff | MLM wrapper | Base model | Median saved | Peak delta saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| opening | 28 | yes | 0 | 28.776 ms | 26.279 ms | 2.497 ms | 0.745 MiB |
| medium | 37 | yes | 0 | 29.726 ms | 28.881 ms | 0.845 ms | 0.985 MiB |
| long | 59 | yes | 0 | 24.991 ms | 21.476 ms | 3.515 ms | 1.571 MiB |

另一次 20-repeat 预跑的绝对耗时约 9.7-10.4 ms，收益为 `-0.045 / 0.438 / 0.687 ms`。两轮都保持 exact parity，但绝对延迟差异说明 Windows 独立测试受 GPU 时钟或系统负载影响，不能把 0.8-3.5 ms 直接承诺为 Linux 端到端收益。

正确性结论不依赖计时：wrapper 的 hidden state 来自同一个 base encoder，跳过的只有无消费者的 vocabulary logits。phone-level feature 仍使用相同的 `word2ph` 索引展开，因此 hidden tensor bitwise equal 会传递为最终 BERT feature bitwise equal。

回归保护：

- 新增测试，若代码再次调用 MLM wrapper 会立即失败。
- 保持 tokenizer、模型加载类型、hidden layer 和 feature expansion 不变。
- Stage 3 完整测试目标更新为 `35 passed, 1 skipped`。
