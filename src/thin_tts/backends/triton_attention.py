"""Triton single-token attention used by the Linux CUDA Graph backend."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dynamic_attention_kernel(
    q_ptr,
    new_k_ptr,
    new_v_ptr,
    k_ptr,
    v_ptr,
    seq_len_ptr,
    out_ptr,
    stride_qh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_oh: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
):
    head = tl.program_id(0)
    offsets_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + head * stride_qh + offsets_d).to(tl.float32)
    valid_len = tl.minimum(tl.load(seq_len_ptr), MAX_SEQ_LEN)

    write_offset = (valid_len - 1) * stride_ks + head * stride_kh + offsets_d
    new_k = tl.load(new_k_ptr + head * stride_qh + offsets_d)
    new_v = tl.load(new_v_ptr + head * stride_qh + offsets_d)
    tl.store(k_ptr + write_offset, new_k)
    tl.store(v_ptr + write_offset, new_v)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # valid_len is a device scalar, so one captured graph skips padded work.
    for start_n in range(0, valid_len, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        valid = offsets_n < valid_len
        matrix_offsets = offsets_n[:, None] * stride_ks + head * stride_kh + offsets_d[None, :]
        k = tl.load(k_ptr + matrix_offsets, mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max)
        block_sum = tl.sum(probabilities, axis=0)

        v_offsets = offsets_n[:, None] * stride_vs + head * stride_vh + offsets_d[None, :]
        v = tl.load(v_ptr + v_offsets, mask=valid[:, None], other=0.0).to(tl.float32)
        accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * v, axis=0)
        running_sum = running_sum * old_scale + block_sum
        running_max = next_max

    tl.store(out_ptr + head * stride_oh + offsets_d, accumulator / running_sum)


def dynamic_attention_with_kv_append(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    block_n: int = 128,
) -> torch.Tensor:
    """Append K/V at ``seq_len - 1`` and attend through ``seq_len``."""
    if new_k.shape != q.shape or new_v.shape != q.shape:
        raise ValueError("new_k and new_v must match q [H,D]")
    if not (q.is_contiguous() and new_k.is_contiguous() and new_v.is_contiguous()):
        raise ValueError("q, new_k, and new_v must be contiguous")
    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and seq_len.is_cuda):
        raise ValueError("all tensors must be CUDA tensors")
    if q.ndim != 2 or k_cache.ndim != 3 or v_cache.shape != k_cache.shape:
        raise ValueError("expected q [H,D] and matching caches [MAX_S,H,D]")
    heads, head_dim = q.shape
    if k_cache.shape[1:] != (heads, head_dim):
        raise ValueError("cache head dimensions do not match q")
    if head_dim not in (16, 32, 64, 128):
        raise ValueError("head dimension must be one of 16, 32, 64, 128")
    if seq_len.dtype != torch.int32 or seq_len.numel() != 1:
        raise ValueError("seq_len must be a one-element CUDA int32 tensor")

    output = torch.empty_like(q)
    _dynamic_attention_kernel[(heads,)](
        q,
        new_k,
        new_v,
        k_cache,
        v_cache,
        seq_len,
        output,
        q.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        output.stride(0),
        HEAD_DIM=head_dim,
        MAX_SEQ_LEN=k_cache.shape[0],
        BLOCK_N=block_n,
        SCALE=1.0 / math.sqrt(head_dim),
        num_warps=4,
    )
    return output
