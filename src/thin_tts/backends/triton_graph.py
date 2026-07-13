"""Production Linux/Triton backend with reusable CUDA Graph and SDPA fallback."""

from __future__ import annotations

import types
from typing import Callable

import torch
import torch.nn.functional as F

from thin_tts.backends.triton_attention import dynamic_attention_with_kv_append


MAX_SEQ_LEN = 1536


class TritonGraphBackend:
    def __init__(
        self,
        model,
        *,
        requested: str = "triton",
        graph_warmup: int = 3,
        strict: bool = False,
        event_sink: Callable[[dict], None] = lambda _event: None,
    ):
        self.model = model
        self.requested = requested
        self.transformer = model.t2s_transformer
        self.blocks = list(self.transformer.blocks)
        if not self.blocks:
            raise ValueError("T2S transformer has no blocks")
        self.original_transformer_prompt = self.transformer.process_prompt
        self.original_transformer_decode = self.transformer.decode_next_token
        self.original_block_decode = [block.decode_next_token for block in self.blocks]
        self.persistent_caches = [(None, None) for _ in self.blocks]
        device = model.ar_audio_embedding.weight.device
        self.shared_seq_len = torch.zeros(1, device=device, dtype=torch.int32)
        self.graph_warmup = max(int(graph_warmup), 0)
        self.strict = bool(strict)
        self.event_sink = event_sink
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.graph_memory_mb = None
        self.capture_count = 0
        self.installed = False
        self._state = "created"
        self._fallback_reason = None
        self._validate_model()

    def _validate_model(self) -> None:
        if self.model.ar_audio_embedding.weight.dtype != torch.float16:
            raise ValueError("Triton graph backend requires FP16 T2S weights")
        for block in self.blocks:
            if block.hidden_dim % block.num_heads:
                raise ValueError("hidden dimension must be divisible by attention heads")
            head_dim = block.hidden_dim // block.num_heads
            if head_dim not in (16, 32, 64, 128):
                raise ValueError(f"unsupported attention head dimension: {head_dim}")

    def status_dict(self) -> dict:
        active = "triton" if self._state in {"created", "installed", "captured"} else "sdpa"
        return {
            "requested": self.requested,
            "active": active,
            "state": self._state,
            "fallback_reason": self._fallback_reason,
            "capture_count": self.capture_count,
            "graph_memory_mb": self.graph_memory_mb,
        }

    def _emit(self, event: str, **fields) -> None:
        self.event_sink({"event": event, **self.status_dict(), **fields})

    def _persistent_prompt_block(self, index, block, x, attn_mask, padding_mask, torch_sdpa):
        q, k, v = F.linear(block.to_mask(x, padding_mask), block.qkv_w, block.qkv_b).chunk(3, dim=-1)
        batch_size, q_len, _ = q.shape
        kv_len = k.shape[1]
        if batch_size != 1:
            raise ValueError(f"Triton graph backend requires batch size 1, got {batch_size}")
        if kv_len >= MAX_SEQ_LEN:
            raise ValueError(f"prompt KV length {kv_len} exceeds capacity {MAX_SEQ_LEN - 1}")

        q = block.to_mask(q, padding_mask)
        k = block.to_mask(k, padding_mask)
        v = block.to_mask(v, padding_mask)
        persistent_k, persistent_v = self.persistent_caches[index]
        expected_shape = (batch_size, MAX_SEQ_LEN, block.hidden_dim)
        if persistent_k is None:
            persistent_k = torch.empty(expected_shape, dtype=k.dtype, device=k.device)
            persistent_v = torch.empty(expected_shape, dtype=v.dtype, device=v.device)
            self.persistent_caches[index] = (persistent_k, persistent_v)
        elif persistent_k.shape != expected_shape or persistent_k.dtype != k.dtype or persistent_k.device != k.device:
            raise ValueError("persistent KV cache shape, dtype, or device changed")
        persistent_k[:, :kv_len].copy_(k)
        persistent_v[:, :kv_len].copy_(v)
        block._k_cache = persistent_k
        block._v_cache = persistent_v
        block._seq_len = kv_len

        q = q.view(batch_size, q_len, block.num_heads, -1).transpose(1, 2)
        k_view = persistent_k[:, :kv_len].view(batch_size, kv_len, block.num_heads, -1).transpose(1, 2)
        v_view = persistent_v[:, :kv_len].view(batch_size, kv_len, block.num_heads, -1).transpose(1, 2)
        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k_view, v_view, ~attn_mask)
        else:
            from thin_tts.models.t2s_model import scaled_dot_product_attention

            attn = scaled_dot_product_attention(q, k_view, v_view, attn_mask)
        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(block.to_mask(attn, padding_mask), block.out_w, block.out_b)
        x = x + attn
        x = F.layer_norm(x, [block.hidden_dim], block.norm_w1, block.norm_b1, block.norm_eps1)
        x = x + block.mlp.forward(x)
        return F.layer_norm(x, [block.hidden_dim], block.norm_w2, block.norm_b2, block.norm_eps2)

    def _graph_prompt(self, transformer, x, attn_mask, padding_mask=None, torch_sdpa=True):
        original_x = x
        try:
            k_cache, v_cache = [], []
            for index, block in enumerate(self.blocks):
                x = self._persistent_prompt_block(
                    index, block, x, attn_mask, padding_mask, torch_sdpa
                )
                k_cache.append(block._k_cache)
                v_cache.append(block._v_cache)
            return x, k_cache, v_cache
        except Exception as exc:
            return self._fallback_prompt(exc, original_x, attn_mask, padding_mask, torch_sdpa)

    @staticmethod
    def _block_decode(block, x, k_cache, v_cache, attn_mask=None, torch_sdpa=True):
        q, new_k, new_v = F.linear(x, block.qkv_w, block.qkv_b).chunk(3, dim=-1)
        if q.shape[:2] != (1, 1):
            raise ValueError(f"Triton graph decode requires [1,1,D] input, got {tuple(q.shape)}")
        head_dim = block.hidden_dim // block.num_heads
        q = q.reshape(block.num_heads, head_dim).contiguous()
        new_k = new_k.reshape(block.num_heads, head_dim).contiguous()
        new_v = new_v.reshape(block.num_heads, head_dim).contiguous()
        cache_shape = (-1, block.num_heads, head_dim)
        attn = dynamic_attention_with_kv_append(
            q,
            new_k,
            new_v,
            block._k_cache.reshape(cache_shape),
            block._v_cache.reshape(cache_shape),
            block._thin_graph_seq_len,
        )
        attn = F.linear(attn.reshape(1, 1, block.hidden_dim), block.out_w, block.out_b)
        x = x + attn
        x = F.layer_norm(x, [block.hidden_dim], block.norm_w1, block.norm_b1, block.norm_eps1)
        x = x + block.mlp.forward(x)
        x = F.layer_norm(x, [block.hidden_dim], block.norm_w2, block.norm_b2, block.norm_eps2)
        return x, block._k_cache, block._v_cache

    def _prepare_length(self):
        old_lengths = [block._seq_len for block in self.blocks]
        next_len = old_lengths[0] + 1
        if next_len >= MAX_SEQ_LEN:
            raise ValueError(f"decode KV length {next_len} exceeds capacity {MAX_SEQ_LEN - 1}")
        if any(length != old_lengths[0] for length in old_lengths):
            raise RuntimeError("T2S block sequence lengths diverged")
        for block in self.blocks:
            block._seq_len = next_len
        self.shared_seq_len.fill_(next_len)
        return old_lengths

    def _capture(self, x, k_cache, v_cache, attn_mask, torch_sdpa):
        self.static_input = torch.empty_like(x)
        self.static_input.copy_(x)
        for _ in range(self.graph_warmup):
            self.original_transformer_decode(
                self.static_input, k_cache, v_cache, attn_mask, torch_sdpa
            )
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output, _, _ = self.original_transformer_decode(
                self.static_input, k_cache, v_cache, attn_mask, torch_sdpa
            )
        torch.cuda.synchronize()
        self.graph = graph
        self.static_output = static_output
        self.graph_memory_mb = round(
            (torch.cuda.memory_allocated() - memory_before) / (1024 * 1024), 2
        )
        self.capture_count += 1
        self._state = "captured"
        self._emit("thin_tts_backend_captured")
        return self.static_output, k_cache, v_cache

    def _graph_decode(self, transformer, x, k_cache, v_cache, attn_mask=None, torch_sdpa=True):
        old_lengths = [block._seq_len for block in self.blocks]
        try:
            if attn_mask is not None:
                raise ValueError("dynamic graph decode expects no attention mask")
            old_lengths = self._prepare_length()
            if self.graph is None:
                return self._capture(x, k_cache, v_cache, attn_mask, torch_sdpa)
            if x.shape != self.static_input.shape or x.dtype != self.static_input.dtype:
                raise ValueError("decode input shape or dtype changed after graph capture")
            self.static_input.copy_(x)
            self.graph.replay()
            return self.static_output, k_cache, v_cache
        except Exception as exc:
            return self._fallback_decode(
                exc, old_lengths, x, k_cache, v_cache, attn_mask, torch_sdpa
            )

    def _mark_fallback(self, exc: Exception) -> None:
        self._state = "fallback"
        self._fallback_reason = f"{type(exc).__name__}: {exc}"
        self.restore(preserve_state=True)
        self.graph = None
        self.static_input = None
        self.static_output = None
        self._emit("thin_tts_backend_fallback")

    def _fallback_prompt(self, exc, x, attn_mask, padding_mask, torch_sdpa):
        if self.strict:
            raise exc
        self._mark_fallback(exc)
        return self.original_transformer_prompt(x, attn_mask, padding_mask, torch_sdpa)

    def _fallback_decode(self, exc, old_lengths, x, k_cache, v_cache, attn_mask, torch_sdpa):
        for block, old_len in zip(self.blocks, old_lengths):
            block._seq_len = old_len
        if self.strict:
            raise exc
        self._mark_fallback(exc)
        return self.original_transformer_decode(x, k_cache, v_cache, attn_mask, torch_sdpa)

    def install(self):
        if self.installed:
            return
        for block in self.blocks:
            block._thin_graph_seq_len = self.shared_seq_len
            block.decode_next_token = types.MethodType(self._block_decode, block)
        self.transformer.process_prompt = types.MethodType(self._graph_prompt, self.transformer)
        self.transformer.decode_next_token = types.MethodType(self._graph_decode, self.transformer)
        self.installed = True
        self._state = "installed"

    def restore(self, preserve_state: bool = False):
        if self.installed:
            self.transformer.process_prompt = self.original_transformer_prompt
            self.transformer.decode_next_token = self.original_transformer_decode
            for block, decode in zip(self.blocks, self.original_block_decode):
                block.decode_next_token = decode
            self.installed = False
        if not preserve_state:
            self._state = "disabled"
