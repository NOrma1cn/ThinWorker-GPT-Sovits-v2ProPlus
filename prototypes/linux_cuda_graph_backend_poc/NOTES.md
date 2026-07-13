# Dynamic Linux backend POC verdict

## Verdict

The design is feasible. One captured 24-layer CUDA Graph successfully reused a
CUDA `int32` scalar for every tested KV length without recapture. The Triton
kernel also fuses the current token's K/V cache write with attention.

Test environment: RTX 4080 Laptop GPU, WSL2, PyTorch 2.12.1+cu126, Triton 3.7.1,
24-layer DPO checkpoint with 16 heads x 32 dimensions.

## Isolated attention

Median latency in microseconds (`BLOCK_N=128`, 2,000 iterations x 5 repeats):

| Valid KV | PyTorch SDPA eager | Triton eager | Triton graph | Graph / SDPA |
|---:|---:|---:|---:|---:|
| 84 | 28.04 | 22.30 | 7.37 | 3.80x |
| 132 | 27.23 | 22.52 | 7.82 | 3.48x |
| 235 | 26.06 | 20.65 | 8.65 | 3.01x |
| 384 | 30.74 | 21.62 | 10.06 | 3.06x |
| 512 | 30.15 | 20.90 | 9.82 | 3.07x |
| 768 | 30.26 | 22.05 | 14.23 | 2.13x |
| 1536 | 30.14 | 23.09 | 22.76 | 1.32x |

Across lengths 1, 17, 84, 132, 235, 384, 512, and 768, maximum FP16 error
against SDPA was 0.000977. Poisoning every cache element after `seq_len` with
NaN did not affect output. A graph captured once at length 1 replayed correctly
at lengths 17, 235, 768, and 84.

## Complete 24-layer decode graph

One dynamic graph, captured once and replayed at all lengths:

| KV prefix | Dynamic graph | Fixed-length graph (same environment) | Dynamic / fixed |
|---:|---:|---:|---:|
| 84 | 0.852 ms | 1.103 ms | 1.29x |
| 132 | 0.876 ms | 1.223 ms | 1.40x |
| 235 | 0.889 ms | 1.081 ms | 1.22x |
| 384 | 0.945 ms | 1.060 ms | 1.12x |
| 512 | 1.006 ms | 1.066 ms | 1.06x |
| 768 | 1.118 ms | 1.163 ms | 1.04x |

The dynamic graph allocated about 8.36 MB. A single fixed-length graph allocated
about 16.25 MB and would require a separate graph for every exact length.

Against SDPA, accumulated 24-layer logits differed by at most 0.0293. All six
representative points retained the same argmax and the same complete Top-5 set;
maximum probability difference was below 0.001. Graph replay matched Triton
eager exactly.

## What remains before production

The POC monkey-patches model blocks only in memory. Production integration still
needs persistent cache addresses, a device-side sequence-length state, static
graph input/output buffers, capture lifecycle handling, and a safe fallback to
SDPA. End-to-end autoregressive audio A/B testing is required because small
per-step numerical differences can eventually change a sampled token.

Recommended next gate: integrate behind an opt-in Linux/Triton backend flag,
retain SDPA as fallback, then generate the existing listening suite before
making it the default.

## End-to-end Mode 4 first-audio result

The reusable runtime was then connected to the real streaming pipeline. The
model, prompt/VITS path, Triton kernel, and graph were warmed before measurement.
Five repeats used counterbalanced backend order and measured from `pipeline.run()`
entry to the first returned PCM chunk:

| Text | SDPA TTFB | Full graph TTFB | Saved | Improvement |
|---|---:|---:|---:|---:|
| Opening / early mute boundary | 216.6 ms | 154.8 ms | 61.8 ms | 28.5% |
| Long stress / fixed 50-token deadline | 271.3 ms | 187.7 ms | 83.6 ms | 30.8% |

Median complete generation fell from 1,135 to 508 ms for the opening text and
from 3,339 to 1,534 ms for the long stress text (about 55% and 54%). The graph
was captured exactly once and allocated 8.13 MB.

All five opening outputs were byte-identical between SDPA and the graph. The
long text was deterministic within each backend but followed the same small
sampling branch already covered by the accepted listening test. Results are in
`eval_output/linux_triton_quality_ab/graph_ttfb_results.json`.
