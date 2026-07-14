"""Measure the upper bound of an exact-shape VITS flow+decoder CUDA Graph."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from validate_vits_bucket_padding import comparison, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("eval_config_wsl_rng.yaml"))
    parser.add_argument("--frames", type=int, default=104)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def elapsed_ms(call, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model = load_model(args.config)
    torch.manual_seed(314159)
    z_p = torch.randn(
        1,
        model.inter_channels,
        args.frames,
        device="cuda",
        dtype=torch.float16,
    )
    mask = torch.ones(1, 1, args.frames, device="cuda", dtype=torch.float16)
    ge = torch.randn(1, model.gin_channels, 1, device="cuda", dtype=torch.float16)

    def eager():
        z = model.flow(z_p, mask, g=ge, reverse=True)
        return model.dec(z * mask, g=ge)

    for _ in range(5):
        eager()
    torch.cuda.synchronize()
    eager_output = eager()

    static_z_p = z_p.clone()
    static_mask = mask.clone()
    static_ge = ge.clone()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(5):
            static_z = model.flow(static_z_p, static_mask, g=static_ge, reverse=True)
            static_output = model.dec(static_z * static_mask, g=static_ge)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    del static_z, static_output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_before, _ = torch.cuda.mem_get_info()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_z = model.flow(static_z_p, static_mask, g=static_ge, reverse=True)
        static_output = model.dec(static_z * static_mask, g=static_ge)
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()

    static_z_p.copy_(z_p)
    static_mask.copy_(mask)
    static_ge.copy_(ge)
    graph.replay()
    torch.cuda.synchronize()
    graph_output = static_output.clone()

    eager_ms = elapsed_ms(eager, args.repeats)

    def replay():
        static_z_p.copy_(z_p)
        static_mask.copy_(mask)
        static_ge.copy_(ge)
        graph.replay()
        return static_output.clone()

    graph_ms = elapsed_ms(replay, args.repeats)
    result = {
        "frames": args.frames,
        "repeats": args.repeats,
        "eager_ms": round(eager_ms, 3),
        "graph_ms": round(graph_ms, 3),
        "improvement_percent": round((eager_ms - graph_ms) / eager_ms * 100, 2),
        "graph_memory_mb": round((free_before - free_after) / 1024**2, 2),
        "output": comparison(eager_output, graph_output),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
