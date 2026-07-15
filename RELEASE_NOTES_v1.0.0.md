# thin-tts-server v1.0.0

首个稳定 API 版本，面向 Windows 11 + RTX 40 系列 GPU，并以 WSL2 Linux
作为 Triton Full Graph 正式运行环境。

## 稳定合同

- 唯一合成接口为 `POST /stream`，固定音色由服务启动配置提供。
- 默认 `mode=4`：首块/低缓冲使用 50-token 上限，缓冲充足后等待自然静音边界。
- 默认 `rng_isolation=true`、`cache_vits_encoded_text=true`。
- 请求模式只接受 2、3、4；未知字段返回 422，不再静默忽略。
- 显式 `hybrid_steady_tokens=0/120` 可回退旧 `50/50`、`50/120` 调度。

## 性能与质量

- Linux Triton Full CUDA Graph 正式后端，失败自动回退 PyTorch SDPA。
- G2PW CUDA Provider 正式链路，失败自动回退 CPU。
- fixed-voice profile 可跳过 CN-HuBERT 与 ERes2Net 启动加载，增加约 252 MiB 可用显存。
- VITS target-text 编码缓存使 long 端到端改善约 5.7%。
- buffer-aware Mode 4 在验证 long 中将 16 块降为 3 块、强制接缝降为 0，总耗时中位数降低约 11.9%。
- 修复 final chunk 爆音、PCM16 正满幅环绕、semantic lookahead 重复等质量问题。
- short/medium/long 多轮 semantic hash、缓冲余量和人工试听通过；用户确认后半段更稳定、失真更少且更干净。

## 部署要求

- Windows 11 主机、WSL2、RTX 40 系列 GPU、支持 WSL CUDA 的 NVIDIA 驱动。
- WSL 内安装兼容驱动的 CUDA 版 PyTorch。
- `onnxruntime-gpu==1.23.2`；不要同时安装 CPU `onnxruntime` 包。
- 模型权重、参考音频和自动下载资产不包含在 wheel 中，需分别遵守其许可证。

完整变更和验证数据见 `CHANGES.md` 与
`docs/inference-audit-and-validation.md`。
