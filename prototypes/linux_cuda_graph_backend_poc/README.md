# PROTOTYPE — Dynamic T2S CUDA Graph Backend

This isolated WSL2/Triton prototype tests whether the 24-layer T2S single-token
decode can use one CUDA Graph while the valid KV length changes on every token.
It does not modify or get imported by production code.

The tested DPO checkpoint uses this decode shape:

- batch 1, query length 1
- hidden size 512
- 16 attention heads, head dimension 32
- FP16 Q/K/V
- preallocated KV capacity 1536

## Files

- `triton_dynamic_attention.py`: device-scalar attention and fused KV append.
- `test_triton_dynamic_attention.py`: numerical, poisoned-tail, fused-append,
  and cross-length graph tests.
- `benchmark_triton_dynamic_attention.py`: isolated attention benchmark.
- `triton_graph_runtime.py`: reusable full-graph runtime with persistent KV addresses.
- `NOTES.md`: measured results and production verdict.

## Run in Ubuntu/WSL2

```bash
cd <repo>/prototypes/linux_cuda_graph_backend_poc
source /home/norma1/.venvs/thin-tts/bin/activate
python test_triton_dynamic_attention.py
python benchmark_triton_dynamic_attention.py
```

Results are written under `eval_output/linux_cuda_graph_backend_poc/`.
