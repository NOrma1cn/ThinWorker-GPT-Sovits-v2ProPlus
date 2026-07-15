"""Validate fixed-voice cache parity and VRAM released by unloading encoders."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml


TEXT = "嗯？你是在叫我吗？嘿嘿，有什么有趣的事情要告诉我呀？"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("eval_config_wsl_rng.yaml"))
    return parser.parse_args()


def cuda_memory() -> dict:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "free_mb": round(free / 1024**2, 2),
        "total_mb": round(total / 1024**2, 2),
    }


def synthesize(pipeline, request: dict) -> dict:
    started = time.perf_counter()
    fragments = [np.asarray(audio).reshape(-1) for _, audio in pipeline.run(request)]
    pcm = np.concatenate(fragments)
    return {
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "samples": int(pcm.size),
        "dtype": str(pcm.dtype),
        "sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    server = raw["server"]
    weights = raw["weights"]
    os.environ["bert_path"] = weights["bert_path"]
    os.environ["THIN_TTS_SV_PATH"] = weights["sv_path"]
    os.environ.setdefault("THIN_TTS_DISABLE_TQDM", "1")

    from thin_tts.g2pw_backend import configure_g2pw_backend, cuda_device_id_from_device
    from thin_tts.pipeline.tts import TTS, TTS_Config

    configure_g2pw_backend(
        requested=server["g2pw_backend"],
        cuda_device_id=cuda_device_id_from_device(server["device"]),
        cuda_memory_limit_mb=server.get("g2pw_cuda_memory_limit_mb"),
    )
    pipeline_config = {
        "custom": {
            "device": server["device"],
            "is_half": server["half"],
            "version": "v2ProPlus",
            "t2s_weights_path": weights["t2s_weights"],
            "vits_weights_path": weights.get("vits_lora") or weights["vits_weights"],
            "bert_base_path": weights["bert_path"],
            "cnhuhbert_base_path": weights["hubert_path"],
            "sv_path": weights["sv_path"],
            "t2s_backend": server["t2s_backend"],
        }
    }
    pipeline = TTS(TTS_Config(pipeline_config))
    request = {
        "text": TEXT,
        "text_lang": "zh",
        "ref_audio_path": weights["ref_audio"],
        "prompt_text": weights["ref_text"],
        "prompt_lang": "zh",
        "text_split_method": "cut2",
        "batch_size": 1,
        "speed_factor": 1.0,
        "streaming_mode": True,
        "parallel_infer": False,
        "split_bucket": False,
        "fixed_length_chunk": False,
        "top_k": 15,
        "top_p": 0.6,
        "temperature": 0.6,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "overlap_length": 2,
        "min_chunk_length": 10,
        "hybrid_switch_tokens": 50,
        "hybrid_steady_tokens": 50,
        "fragment_interval": 0.0,
        "seed": 314159,
        "rng_isolation": True,
        "cache_vits_encoded_text": True,
    }

    synthesize(pipeline, request)
    baseline = synthesize(pipeline, request)
    memory_before = cuda_memory()
    cache_state = {
        "prompt_semantic": pipeline.prompt_cache["prompt_semantic"] is not None,
        "refer_spec": len(pipeline.prompt_cache["refer_spec"]),
        "sv_embeddings": len(pipeline.prompt_cache.get("sv_emb_list") or []),
        "prompt_phones": pipeline.prompt_cache["phones"] is not None,
        "prompt_bert": pipeline.prompt_cache["bert_features"] is not None,
    }

    hubert = pipeline.cnhuhbert_model
    speaker = pipeline.sv_model
    pipeline.cnhuhbert_model = None
    pipeline.sv_model = None
    del hubert, speaker
    gc.collect()
    torch.cuda.empty_cache()
    memory_after = cuda_memory()
    unloaded = synthesize(pipeline, request)

    result = {
        "cache_state": cache_state,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "free_memory_gain_mb": round(
            memory_after["free_mb"] - memory_before["free_mb"],
            2,
        ),
        "baseline": baseline,
        "after_unload": unloaded,
        "pcm_bitwise_equal": baseline["sha256"] == unloaded["sha256"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
