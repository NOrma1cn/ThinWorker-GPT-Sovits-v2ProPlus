"""POC-only reusable full-transformer CUDA Graph backend."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from triton_dynamic_attention import dynamic_attention_with_kv_append


class TritonGraphBackend:
    """Switch a loaded T2S model between original SDPA and one dynamic graph."""

    def __init__(self, model, graph_warmup: int = 3):
        self.model = model
        self.transformer = model.t2s_transformer
        self.blocks = list(self.transformer.blocks)
        self.original_transformer_decode = self.transformer.decode_next_token
        self.original_block_decode = [block.decode_next_token for block in self.blocks]
        self.original_process_prompt = [block.process_prompt for block in self.blocks]
        self.persistent_caches = [(None, None) for _ in self.blocks]
        self.shared_seq_len = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.graph_warmup = graph_warmup
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.graph_memory_mb = None
        self.capture_count = 0
        self.installed = False

    def _persistent_prompt(self, index, block, x, attn_mask, padding_mask=None, torch_sdpa=True):
        output, new_k, new_v = self.original_process_prompt[index](
            x, attn_mask, padding_mask, torch_sdpa
        )
        persistent_k, persistent_v = self.persistent_caches[index]
        if persistent_k is None:
            persistent_k, persistent_v = new_k, new_v
            self.persistent_caches[index] = (persistent_k, persistent_v)
        elif new_k.data_ptr() != persistent_k.data_ptr():
            valid_len = block._seq_len
            persistent_k[:, :valid_len].copy_(new_k[:, :valid_len])
            persistent_v[:, :valid_len].copy_(new_v[:, :valid_len])
        block._k_cache = persistent_k
        block._v_cache = persistent_v
        return output, persistent_k, persistent_v

    @staticmethod
    def _block_decode(block, x, k_cache, v_cache, attn_mask=None, torch_sdpa=True):
        q, new_k, new_v = F.linear(x, block.qkv_w, block.qkv_b).chunk(3, dim=-1)
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
        next_len = self.blocks[0]._seq_len + 1
        for block in self.blocks:
            if block._seq_len + 1 != next_len:
                raise RuntimeError("T2S block sequence lengths diverged")
            block._seq_len = next_len
        self.shared_seq_len.fill_(next_len)

    def _capture(self, x, k_cache, v_cache, attn_mask, torch_sdpa):
        self.static_input = torch.empty_like(x)
        self.static_input.copy_(x)
        for _ in range(self.graph_warmup):
            self.original_transformer_decode(
                self.static_input, k_cache, v_cache, attn_mask, torch_sdpa
            )
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output, _, _ = self.original_transformer_decode(
                self.static_input, k_cache, v_cache, attn_mask, torch_sdpa
            )
        torch.cuda.synchronize()
        self.graph_memory_mb = round(
            (torch.cuda.memory_allocated() - memory_before) / (1024 * 1024), 2
        )
        self.capture_count += 1
        return self.static_output, k_cache, v_cache

    def _graph_decode(self, transformer, x, k_cache, v_cache, attn_mask=None, torch_sdpa=True):
        if attn_mask is not None:
            raise ValueError("dynamic graph decode expects no attention mask")
        self._prepare_length()
        if self.graph is None:
            return self._capture(x, k_cache, v_cache, attn_mask, torch_sdpa)
        if x.shape != self.static_input.shape or x.dtype != self.static_input.dtype:
            raise ValueError("decode input shape or dtype changed after graph capture")
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output, k_cache, v_cache

    def install(self):
        if self.installed:
            return
        backend = self
        for index, block in enumerate(self.blocks):
            block._thin_graph_seq_len = self.shared_seq_len

            def process_prompt(this, x, attn_mask, padding_mask=None, torch_sdpa=True, *, _index=index):
                return backend._persistent_prompt(
                    _index, this, x, attn_mask, padding_mask, torch_sdpa
                )

            block.process_prompt = types.MethodType(process_prompt, block)
            block.decode_next_token = types.MethodType(self._block_decode, block)
        self.transformer.decode_next_token = types.MethodType(self._graph_decode, self.transformer)
        self.installed = True

    def restore(self):
        if not self.installed:
            return
        self.transformer.decode_next_token = self.original_transformer_decode
        for block, decode, prompt in zip(
            self.blocks, self.original_block_decode, self.original_process_prompt
        ):
            block.decode_next_token = decode
            block.process_prompt = prompt
        self.installed = False
