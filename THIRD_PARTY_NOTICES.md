# Third-party notices

thin-tts-server is a trimmed inference distribution derived primarily from
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Source comments retain
more precise provenance where individual implementations were adapted.

The following upstream projects are represented in, or referenced by, the
distributed source. Their copyrights remain with their respective owners.

| Project | License | Use/provenance |
|---|---|---|
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | MIT, Copyright (c) 2024 RVC-Boss | Primary upstream inference implementation |
| [VALL-E](https://github.com/lifeiteng/vall-e) | Apache-2.0 | Transformer, embedding, activation and AR references |
| [SoundStorm](https://github.com/yangdongchao/SoundStorm) | No standalone license file detected upstream; SoundStorm-derived files entered this distribution through GPT-SoVITS's MIT-licensed source and also cite VALL-E | AR model provenance |
| [3D-Speaker](https://github.com/modelscope/3D-Speaker) | Apache-2.0 | ERes2Net speaker encoder |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | Apache-2.0 | Speaker pooling implementation reference |
| [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) | Apache-2.0 | Chinese normalization and G2PW frontend adaptations |
| [g2pW](https://github.com/GitYCC/g2pW) | Apache-2.0 | G2PW dataset/runtime adaptations |
| [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW) | MIT, Copyright (c) 2022 mozillazg | Python G2PW wrapper adaptation |
| [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) | MIT, Copyright (c) 2020 Phil Wang | Vector quantization implementation reference |
| [UniLM](https://github.com/microsoft/unilm) | MIT, Copyright (c) Microsoft Corporation | Sampling utility reference |
| [Lightning Bolts](https://github.com/Lightning-Universe/lightning-bolts) | Apache-2.0 | Distributed utility reference |
| [Textual](https://github.com/Textualize/textual) | MIT | Terminal user interface framework |
| [ruamel.yaml](https://sourceforge.net/projects/ruamel-yaml/) | MIT | Comment-preserving YAML configuration editing |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | Detached process discovery and lifecycle management |
| [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) | Apache-2.0 | Runtime model metric dependency |

The project MIT license is in `LICENSE`. The Apache License 2.0 text applying
to Apache-licensed portions is in `LICENSES/Apache-2.0.txt`. Upstream model
weights and downloaded model assets may have separate licenses; users are
responsible for reviewing the terms of the particular checkpoints and assets
they distribute.
