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

### 2026-07-14：V4-A Request-local RNG isolation

状态：默认关闭的 POC 通过。不同 chunk 调度不再改变 T2S semantic token；音频仍会随 VITS chunk shape 和拼接变化，尚未通过主观质量门槛。

根因验证：

- 旧链路在请求开始时重置全局 CUDA RNG。
- T2S `exponential_` sampling 与每次 VITS `randn_like` 交替消耗同一个 RNG stream。
- T2S Python generator 每次 yield 后会先运行一个 VITS chunk，再恢复 semantic generation。
- 因此 chunk 长度改变了恢复 T2S 时的 RNG state，调度策略实际也改变了模型生成内容。

POC 实现：

- 为同一 request seed 派生独立的 T2S 和 VITS `torch.Generator`。
- T2S generator 只进入 semantic sampling。
- VITS generator 只进入 streaming latent noise。
- `rng_isolation=false` 为默认值，旧输出不变；当前只通过 API 显式开启实验。
- profile 模式记录累计 semantic tensor 的 SHA-256，不在非 profile 请求中引入 GPU-to-CPU 同步。

Linux 正式链路环境：Triton Full Graph captured、G2PW CUDA、DPO T2S/VITS、seed `314159`。测试文本为 98 个中文字符，固定长度 mode 3 分别使用 40 和 80 semantic-token chunk，每组重复两次。

| RNG | Chunk | Repeat | Final tokens | Final semantic SHA-256 |
|---|---:|---:|---:|---|
| legacy global | 40 | 1/2 | 469 | `82d1c9f0543f...f6d7c53` |
| legacy global | 80 | 1/2 | 440 | `198f3126da43...3524ee` |
| isolated | 40 | 1 | 456 | `fca51b14d498...bf2499` |
| isolated | 40 | 2 | 456 | `fca51b14d498...bf2499` |
| isolated | 80 | 1 | 456 | `fca51b14d498...bf2499` |
| isolated | 80 | 2 | 456 | `fca51b14d498...bf2499` |

结论：旧 global RNG 下，40/80 并不是同一条 T2S 内容的调度对照；隔离后四次最终 semantic tensor bitwise identical，V4 后续的 `first/steady` 调度实验才具备可比性。

隔离后的 PCM 仍不要求跨 chunk schedule 相同：chunk 40 和 80 的 VITS noise tensor shape、累计前缀 decode 和 SOLA 接缝不同。实测两者 WAV 长度相差 108 samples（约 3.4 ms），hash 也不同。这部分必须用边界指标和用户试听判断，不能用 semantic parity 替代。

基准工具最初把立即发送的 44-byte WAV header 误记为音频 TTFB，已修正为累计响应超过 44 bytes 时才记录 `audio_ttfb_ms`。错误的 3-27 ms header 数据不进入结论。

代码级回归：独立 generator、默认关闭、服务请求透传和全局 RNG 干扰测试通过；当前完整测试为 `40 passed, 1 skipped`。

### 2026-07-14：V4-B Mute-boundary correctness and `50/120` scheduler

状态：代码、客观指标和用户试听均通过。semantic lookahead 重复修复进入正式链路；`50/120` 保留为显式实验参数，但因 long 总收益未达门槛，不进入默认 mode 4。

#### 先修复验证中发现的 semantic lookahead 重复

旧 mute-boundary 分支在找到边界后：

```text
yield:    y_buf[curr_ptr:y_len_total]      # 包含边界后的 2-token lookahead
advance:  curr_ptr += argmax_idx + 1       # 只推进到边界
```

因此边界后的 lookahead 已经发送给 VITS，却会在下一 chunk 再发送一次。修复前同一条原始 T2S 序列出现：

| Policy | Raw T2S tokens | Pipeline accumulated tokens | Duplicated |
|---|---:|---:|---:|
| opening `50/50` | 131 | 133 | 2 |
| long `50/50` | 456 | 468 | 12 |
| long `50/120` | 456 | 465 | 9 |

修复后只 yield 到 `argmax_idx + 1`，lookahead 留在 pending buffer。所有 12 个正式请求均满足 raw T2S tokens 等于 pipeline tokens；opening 为 131，long 为 456。该问题会改变音频，性质是现有边界正确性修复，而不是调度性能优化。

#### Adaptive deadline

新增显式 `hybrid_steady_tokens`：

- 首 chunk 仍使用 `hybrid_switch_tokens=50`。
- 后续 chunk 可显式设为 120。
- 未设置或设为 0 时继续使用 50，现有 mode 4 默认行为不变。
- profile event 记录每个 chunk 实际生效的 deadline。

Linux/Triton、G2PW CUDA、seed `314159`、request-local RNG 下，预热后交替运行 `50/50` 和 `50/120`，opening/long 各 3 次。

semantic 正确性：

| Text | Policy | Pipeline tokens | Final semantic SHA-256 |
|---|---|---:|---|
| opening | `50/50` | 131 | `2c3e8e532b12...a577c3` |
| opening | `50/120` | 131 | `2c3e8e532b12...a577c3` |
| long | `50/50` | 456 | `fca51b14d498...bf2499` |
| long | `50/120` | 456 | `fca51b14d498...bf2499` |

实际 chunk 数：

| Text | `50/50` | `50/120` |
|---|---:|---:|
| opening | 5 | 2 个非空 audio chunks（另有 final empty marker） |
| long | 16 | 12 |

服务端同步 profile 中位数：

| Text | Policy | First chunk elapsed | VITS total | Last audio chunk elapsed |
|---|---|---:|---:|---:|
| opening | `50/50` | 87.4 ms | 136.3 ms | 450.4 ms |
| opening | `50/120` | 84.5 ms | 65.5 ms | 382.2 ms |
| long | `50/50` | 178.0 ms | 456.7 ms | 1638.3 ms |
| long | `50/120` | 179.2 ms | 379.5 ms | 1607.8 ms |

客户端中位数：

| Text | Policy | PCM TTFB | Total | Audio duration |
|---|---|---:|---:|---:|
| opening | `50/50` | 95.3 ms | 462.2 ms | 5.2396 s |
| opening | `50/120` | 94.6 ms | 396.3 ms | 5.2400 s |
| long | `50/50` | 186.0 ms | 1647.2 ms | 18.2002 s |
| long | `50/120` | 213.9 ms | 1621.3 ms | 18.2057 s |

客户端 long TTFB 存在 Windows HTTP 抖动；服务端首 chunk profile 显示两策略实际为 `178.0 vs 179.2 ms`，符合“首 chunk 调度不变”。因此不能把客户端 `+27.9 ms` 解释为算法回退。

收益结论：opening 的 VITS 调用数明显下降，总耗时改善约 15%；long VITS 累计改善约 17%，但 T2S 占主导，端到端只改善约 1.9%。这没有达到 V4 原定 long total 至少 10% 的 promotion 门槛。

客观边界指标：两组都没有 clipping。adaptive 的最大单点跳变略高：opening `17096 -> 18864`，long `12578 -> 14368`；p99.9 jump 为 opening `9549 -> 8451`、long `7584 -> 7874`。这些统计不能代替试听，尤其要听更大的 steady chunk 是否带来接缝、重音或语气变化。

用户试听结论：旧 `50/50`、修复重复后的 `50/50`、修复后的 `50/120`，opening 与 long 均未发现质量问题。由此接受 lookahead 正确性修复；adaptive 虽通过质量门槛，但仍因性能门槛失败而不默认启用。

### 2026-07-14：V5-A GE512 cache

状态：拒绝。PCM bitwise 等价通过，但 long total 没有稳定改善；实验实现和 API 开关均未保留。

v2ProPlus 的流式 VITS 已跨 chunk 缓存 GE，但每次仍执行 `ge_to512`。实验增加了显式 `cache_vits_ge512` 开关，只复用第一次计算的 GE512；mode 4 保持 `50/50`，固定 seed `314159` 并启用 request-local RNG。opening/long 各自预热后交替运行基线与缓存组 7 次。

正确性结果：两段文本在所有重复和两种策略间的 WAV SHA-256 分别保持唯一，PCM bitwise equal。

服务端 profile 中位数：

| Text | Policy | First VITS | VITS total | Last chunk elapsed |
|---|---|---:|---:|---:|
| opening | baseline | 35.0 ms | 181.9 ms | 594.3 ms |
| opening | cache GE512 | 34.1 ms | 183.6 ms | 617.0 ms |
| long | baseline | 42.5 ms | 636.8 ms | 2242.3 ms |
| long | cache GE512 | 43.4 ms | 618.3 ms | 2288.4 ms |

客户端中位数：

| Text | Policy | PCM TTFB | Total |
|---|---|---:|---:|
| opening | baseline | 128.7 ms | 614.7 ms |
| opening | cache GE512 | 120.2 ms | 632.6 ms |
| long | baseline | 251.4 ms | 2259.3 ms |
| long | cache GE512 | 253.3 ms | 2306.4 ms |

long VITS 累计中位数表面改善 18.5 ms（约 2.9%），但逐轮配对差值方向不一致；long 服务端端到端逐轮配对中位差为 `+22.7 ms`，客户端为 `+22.4 ms`。该投影只占 VITS 总成本的极小部分，收益落入运行抖动，没有满足“long total 稳定改善”的 V5 门槛，因此不增加正式链路复杂度。

### 2026-07-14：V5-B Encoded target text cache

状态：接受并默认启用，保留 `cache_vits_encoded_text=false` 请求级回退。

`TextEncoder.forward` 原先在每个 VITS chunk 都重复执行固定 phones 的：

```text
sequence mask
-> text embedding
-> encoder_text Transformer
```

正式实现只缓存这三步的 encoded text 与 mask。`encoder_ssl`、MRTE、`encoder2`、latent sampling、flow 和 waveform decoder 仍按 chunk 执行；缓存生命周期限制在单个 pipeline item 内，不跨文本请求复用。非流式和训练调用继续使用原来的 6 元返回接口。

实验保持 mode 4 `50/50`、seed `314159`、request-local RNG；opening/long 各自预热后交替运行基线与缓存组 7 次。所有重复和两种策略间的 WAV SHA-256 唯一，PCM bitwise equal。

服务端 profile 中位数：

| Text | Policy | First VITS | VITS total | Last chunk elapsed |
|---|---|---:|---:|---:|
| opening | baseline | 37.3 ms | 194.3 ms | 704.9 ms |
| opening | cache encoded text | 36.0 ms | 154.3 ms | 661.3 ms |
| long | baseline | 39.9 ms | 585.1 ms | 2351.2 ms |
| long | cache encoded text | 39.3 ms | 466.5 ms | 2217.6 ms |

客户端中位数：

| Text | Policy | PCM TTFB | Total |
|---|---|---:|---:|
| opening | baseline | 131.5 ms | 719.2 ms |
| opening | cache encoded text | 133.0 ms | 678.3 ms |
| long | baseline | 268.3 ms | 2366.8 ms |
| long | cache encoded text | 262.3 ms | 2232.7 ms |

收益来自预期阶段：long VITS 累计中位数改善 118.6 ms（约 20.3%），服务端和客户端 long total 都改善约 5.7%。逐轮配对中，VITS 为 7/7 更快，服务端和客户端 total 均为 6/7 更快；唯一回退轮次仍保持 VITS 更快，端到端差值来自 T2S/系统抖动。opening TTFB 中位数变化 `+1.5 ms`，未形成实质回退。

正式默认 smoke 在请求体不提供缓存字段时记录 `vits_text_cache=true`；同一服务进程内显式 `false` 与默认 `true` 的 opening WAV SHA-256 均为 `8ce97e350680...f401b`，确认默认透传和回退路径都保持 bitwise 等价。

### 2026-07-15：V6 Static-shape VITS graph/buckets

状态：当前方案拒绝。保留 profile-only 的 VITS CUDA Event 分段计时和独立 POC；不引入 graph backend。

#### 正式缓存链路的 VITS 分布

CUDA Event 只在 `THIN_TTS_PROFILE` 下创建，正常请求不增加 GPU event 或同步。正式 encoded-text cache 开启时，3 次重复的阶段累计中位数为：

| Text | VITS total | conditioning | setup | RVQ | enc_p | noise | flow | decoder |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opening | 134.2 ms | 1.6 | 0.6 | 1.4 | 49.5 | 0.5 | 30.2 | 50.9 |
| long | 377.7 ms | 1.9 | 2.6 | 3.7 | 125.1 | 2.3 | 93.9 | 148.9 |

long 中 `enc_p` 约占 33.1%，flow 约 24.9%，waveform decoder 约 39.4%；flow+decoder 合计约 64.3%，从占比上值得验证 graph。

mode 4 的 opening/long 实际 tail latent frames 包含 `10, 22, 26, 28, 30, 36, 38, 44, 54, 70, 84, 86, 100, 104`。首 chunk 的 shape 由 mute boundary 决定，并不等于固定 deadline。

#### 普通 padding bucket 不满足等价门槛

将 representative latent 右侧补零到 `[32, 64, 96, 104]` 后再裁剪有效输出：

- Flow 在部分 shape 上已经不 bitwise equal，观察到最大绝对差约 `0.0059`。
- Waveform decoder 除原生 `104 -> 104` 外全部不等价，最大绝对差约 `0.1144`。
- 多数 padded shape 从 sample 0 就发生差异，不局限于末尾 overlap 保护区。

因此不能把 padding bucket 作为“严格等价的静态 shape”实现；它会变成新的近似质量实验，并可能影响整段音频，而非只影响接缝。

#### Exact-shape CUDA Graph 上限

POC 捕获 flow+decoder，回放计时包含静态输入 copy 和输出 clone；相同 shape 的输出均 bitwise equal：

| Latent frames | Eager | Graph | Local improvement | Graph memory |
|---:|---:|---:|---:|---:|
| 22 | 12.04 ms | 4.13 ms | 65.7% | 未采信早期测量 |
| 70 | 12.33 ms | 7.58 ms | 38.5% | 124 MiB |
| 100 | 13.19 ms | 10.02 ms | 24.0% | 126 MiB |
| 104 | 13.20 ms | 10.44 ms | 21.0% | 126 MiB |

22/70 是当前两条测试文本的首 chunk shape，预先捕获时可使其 first VITS 达到 15% 门槛；但实际文本的 mute-boundary shape 事先未知，为这些 shape 预捕获会过拟合测试集。按需捕获会让第一次请求承担 capture 延迟，并随文本形成 recapture storm。

唯一可由调度策略预知的 exact shapes 是 first force-cut 的 100 和 steady force-cut 的 104。两图约占 252 MiB；100-frame graph 只节省约 3.17 ms，投影到完整 first VITS 低于 15% 门槛。当前 long 样本只有 3 个 104-frame chunk，累计理论收益约 8.3 ms，不到端到端总耗时的 1%。

结论：exact graph 有局部效果，但“少量图、无 recapture storm、首包至少 15%”三个条件无法同时满足。除非未来调度器改为质量可接受的固定 shape，或 decoder 换成原生 chunk-aware/static-shape 结构，否则不推进当前 VITS CUDA Graph backend。

### 2026-07-15：V7 Bounded semantic context

状态：性能门槛失败，未进入主观推广门槛；实验 API 和实现均未保留。

实验保持 mode 4 `50/50`、encoded-text cache、seed `314159` 和 request-local RNG，只将送入 VITS 的累计 semantic 左历史分别限制为 150/100/75 tokens（约 6/4/3 秒）。T2S 仍生成完整 456 tokens，所有策略 final semantic SHA-256 一致。

3 次交替运行的 long 中位数：

| Context | enc_p total | VITS total | Server last chunk | Client total |
|---|---:|---:|---:|---:|
| full | 123.5 ms | 369.1 ms | 1485.3 ms | 1502.9 ms |
| 6 s / 150 | 118.2 ms | 363.6 ms | 1473.4 ms | 1481.7 ms |
| 4 s / 100 | 121.1 ms | 366.6 ms | 1512.6 ms | 1544.1 ms |
| 3 s / 75 | 124.8 ms | 369.1 ms | 1540.8 ms | 1548.8 ms |

6 秒是唯一略快的策略，但只让 `enc_p` 改善 5.3 ms、VITS total 改善约 1.5%、服务端端到端改善约 0.8%；客户端约 1.4% 的变化仍在系统抖动量级。4 秒和 3 秒没有收益。

opening 只有 131 semantic tokens，因此 6 秒策略没有发生截断，WAV 与 full bitwise equal；4/3 秒和所有 long bounded 策略都会改变音频。由于近似方案已经改变音频却没有稳定速度价值，不应再消耗主观质量预算。试听产物仍保留在 `eval_output/vits_context/index.html` 供复核，但不进入正式链路。

### 2026-07-15：V8 Offline fixed-voice profile

状态：接受为显式配置的正式路径。未配置 `voice_profile` 时保持原启动行为；配置后首次编译，后续命中时跳过 CN-HuBERT 与 ERes2Net 加载。

profile schema v1 保存：

- prompt semantic tokens
- reference spectrogram 与 16 kHz reference audio tensor
- speaker embedding list
- prompt phones、BERT features、normalized text 与 language
- reference path 和 auxiliary-reference state

metadata 包含 T2S/VITS/BERT/HuBERT/SV artifact signature、reference audio SHA-256、prompt text/language 和 schema version。reference、prompt 或 checkpoint 常规变更都会得到 `fingerprint_mismatch`；缺失、不可读或 cache 字段不完整会进入完整 warmup 重建，保存使用同目录临时文件加原子 replace。

同进程 POC 在 cache 填满后卸载 CN-HuBERT 与 ERes2Net：

| Metric | Before | After |
|---|---:|---:|
| CUDA allocated | 1461.7 MiB | 1169.6 MiB |
| CUDA reserved | 1544.0 MiB | 1292.0 MiB |
| CUDA free | 8383.0 MiB | 8635.0 MiB |

可用显存增加约 252 MiB；卸载前后均为 167,667 PCM samples，SHA-256 同为 `f4a5c121172c...e7d5`，bitwise equal。

正式两次独立启动验证：第一次 health 为 `compiled`、保存文件 1,084,893 bytes、encoder 已卸载；第二次为 `loaded`，初始化日志不再出现 CN-HuBERT/SV 加载。loaded 服务 warm opening 的 WAV SHA-256 为稳定基线 `8ce97e350680...f401b`，TTFB 约 122 ms。跨进程第一个未覆盖 shape 仍存在此前已观察到的 CUDA cold-shape 数值差异，不由 voice profile 引入；warm 后 profile/no-profile 与回退路径一致。

### 2026-07-15：V9 Buffer-aware Mode 4

状态：客观门槛与用户试听均通过，进入 Mode 4 默认链路。显式 `hybrid_steady_tokens=0/120` 继续提供旧 `50/50`、`50/120` 回退。

旧 Mode 4 对每个 semantic chunk 重复使用 50-token deadline。long 虽有充足的待播放音频，仍被拆成 16 块并执行 15 次 VITS/SOLA handoff；用户指出后半段稳定性较差。新策略保留首块 50-token 上限，之后根据服务端已输出音频时长减去首包后的墙钟耗时估算播放缓冲：低于 500 ms 时启用 50-token 上限，充足时只等待自然静音边界。

最初的逐 token 动态版本会在 GPU 偶发变慢时于同一 semantic chunk 中途改变 deadline，一轮 long 在 195 token 被截断，形成非确定的 5 块输出。因此正式实现只在每个 semantic chunk 开始时采样一次水位，并在块内冻结决定；下一块开始时再评估，兼顾确定切块与低水位保护。

Triton、G2PW CUDA、fixed-voice profile、seed `314159`、request-local RNG 和 profiling 下，旧 `50/50` 与 buffer 500 ms 交替各运行 7 次：

| Text | Legacy chunks | Buffer chunks | Legacy forced | Buffer forced | Minimum predicted margin |
|---|---:|---:|---:|---:|---:|
| short | 2 | 2 | 0 | 0 | 542.4 ms |
| medium | 5 | 3 | 2 | 1 | 132.5 ms |
| long | 16 | 3 | 3 | 0 | 497.9 ms |

| Text | Legacy TTFB | Buffer TTFB | Legacy total | Buffer total |
|---|---:|---:|---:|---:|
| short | 127.2 ms | 134.2 ms | 217.0 ms | 229.2 ms |
| medium | 135.6 ms | 131.4 ms | 570.7 ms | 520.9 ms |
| long | 229.7 ms | 246.4 ms | 1870.0 ms | 1648.0 ms |

两组每个文本的首块 token 数和首块 semantic SHA-256 相同，最终 semantic token 数/hash 也相同。long buffer 路径的跨运行 PCM 差异为 CUDA 大 shape 数值噪声：相关系数约 `0.9999998`、最大 93/32768，不是语义或切块漂移。用户试听确认没有首包后停顿或新增质量问题，并观察到后半段失真减少、声音更干净。
