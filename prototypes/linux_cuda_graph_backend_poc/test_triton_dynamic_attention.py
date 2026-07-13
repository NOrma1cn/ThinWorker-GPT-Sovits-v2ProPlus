from __future__ import annotations

import importlib.util

import torch
import torch.nn.functional as F

if importlib.util.find_spec("triton") is None:
    import pytest

    pytest.skip("Triton POC requires a Linux Triton environment", allow_module_level=True)

from triton_dynamic_attention import dynamic_attention, dynamic_attention_with_kv_append


LENGTHS = (1, 17, 84, 132, 235, 384, 512, 768)
HEADS = 16
HEAD_DIM = 32
MAX_SEQ_LEN = 1536


def reference(q, k, v, length):
    return F.scaled_dot_product_attention(
        q[None, :, None, :],
        k[:length].permute(1, 0, 2)[None],
        v[:length].permute(1, 0, 2)[None],
    )[0, :, 0]


def main():
    torch.manual_seed(20260713)
    device = "cuda"
    q = torch.randn(HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    k = torch.randn(MAX_SEQ_LEN, HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    seq_len = torch.tensor([1], device=device, dtype=torch.int32)

    rows = []
    for length in LENGTHS:
        seq_len.fill_(length)
        actual = dynamic_attention(q, k, v, seq_len)
        expected = reference(q, k, v, length)
        error = (actual - expected).abs()
        rows.append((length, error.max().item(), error.mean().item()))
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

        poisoned_k = k.clone()
        poisoned_v = v.clone()
        poisoned_k[length:] = float("nan")
        poisoned_v[length:] = float("nan")
        poisoned = dynamic_attention(q, poisoned_k, poisoned_v, seq_len)
        torch.testing.assert_close(poisoned, actual, atol=0, rtol=0)

    # Capture once at length 1, then vary only the device scalar. The graph
    # output must follow the new valid prefix without recapture.
    seq_len.fill_(1)
    dynamic_attention(q, k, v, seq_len)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = dynamic_attention(q, k, v, seq_len)
    for length in (17, 235, 768, 84):
        seq_len.fill_(length)
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(q, k, v, length)
        torch.testing.assert_close(graph_output, expected, atol=2e-3, rtol=2e-3)

    new_k = torch.randn_like(q)
    new_v = torch.randn_like(q)
    append_k = k.clone()
    append_v = v.clone()
    seq_len.fill_(235)
    appended_output = dynamic_attention_with_kv_append(
        q, new_k, new_v, append_k, append_v, seq_len
    )
    expected_k = k.clone()
    expected_v = v.clone()
    expected_k[234] = new_k
    expected_v[234] = new_v
    expected = reference(q, expected_k, expected_v, 235)
    torch.testing.assert_close(append_k[234], new_k, atol=0, rtol=0)
    torch.testing.assert_close(append_v[234], new_v, atol=0, rtol=0)
    torch.testing.assert_close(appended_output, expected, atol=2e-3, rtol=2e-3)

    print("numerical, poisoned-tail, fused append, and dynamic CUDA Graph checks passed")
    for length, max_error, mean_error in rows:
        print(f"length={length:4d} max_error={max_error:.6f} mean_error={mean_error:.6f}")


if __name__ == "__main__":
    main()
