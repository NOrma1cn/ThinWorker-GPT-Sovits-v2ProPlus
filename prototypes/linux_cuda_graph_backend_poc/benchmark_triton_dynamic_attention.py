"""Benchmark dynamic-length Triton decode attention against PyTorch SDPA."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from triton_dynamic_attention import dynamic_attention


LENGTHS = (1, 17, 84, 132, 235, 384, 512, 768, 1536)
HEADS = 16
HEAD_DIM = 32
MAX_SEQ_LEN = 1536


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--block-n", type=int, default=128, choices=(16, 32, 64, 128))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "eval_output"
        / "linux_cuda_graph_backend_poc"
        / "triton_dynamic_attention_results.json",
    )
    return parser.parse_args()


def sdpa(q, k, v, length):
    return F.scaled_dot_product_attention(
        q[None, :, None, :],
        k[:length].permute(1, 0, 2)[None],
        v[:length].permute(1, 0, 2)[None],
    )[0, :, 0]


def measure(operation, iterations, repeats):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6 / iterations)
    return statistics.median(samples), samples


def main():
    args = parse_args()
    torch.manual_seed(20260713)
    q = torch.randn(HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    k = torch.randn(MAX_SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    seq_len = torch.tensor([1], device="cuda", dtype=torch.int32)

    # Compile and capture only once. seq_len remains a live device input.
    dynamic_attention(q, k, v, seq_len, args.block_n)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = dynamic_attention(q, k, v, seq_len, args.block_n)

    rows = []
    for length in LENGTHS:
        seq_len.fill_(length)
        for _ in range(args.warmup):
            sdpa(q, k, v, length)
            dynamic_attention(q, k, v, seq_len, args.block_n)
            graph.replay()

        expected = sdpa(q, k, v, length)
        graph.replay()
        torch.cuda.synchronize()
        error = (graph_output - expected).abs()

        sdpa_us, sdpa_samples = measure(lambda: sdpa(q, k, v, length), args.iterations, args.repeats)
        triton_us, triton_samples = measure(
            lambda: dynamic_attention(q, k, v, seq_len, args.block_n), args.iterations, args.repeats
        )
        graph_us, graph_samples = measure(graph.replay, args.iterations, args.repeats)
        row = {
            "seq_len": length,
            "sdpa_eager_us": round(sdpa_us, 3),
            "triton_eager_us": round(triton_us, 3),
            "triton_graph_us": round(graph_us, 3),
            "eager_vs_sdpa": round(sdpa_us / triton_us, 3),
            "graph_vs_sdpa": round(sdpa_us / graph_us, 3),
            "max_abs_error": float(error.max().item()),
            "mean_abs_error": float(error.mean().item()),
            "samples_us": {
                "sdpa": [round(x, 3) for x in sdpa_samples],
                "triton": [round(x, 3) for x in triton_samples],
                "graph": [round(x, 3) for x in graph_samples],
            },
        }
        rows.append(row)
        print(
            f"length={length:4d} sdpa={sdpa_us:7.2f}us triton={triton_us:7.2f}us "
            f"graph={graph_us:7.2f}us graph_speedup={sdpa_us / graph_us:5.2f}x "
            f"error={row['max_abs_error']:.6f}",
            flush=True,
        )

    result = {
        "prototype": "triton_dynamic_seq_len_single_token_attention",
        "shape": {"batch": 1, "heads": HEADS, "head_dim": HEAD_DIM, "max_seq_len": MAX_SEQ_LEN},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "triton": __import__("triton").__version__,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "block_n": args.block_n,
        "single_graph_all_lengths": True,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
