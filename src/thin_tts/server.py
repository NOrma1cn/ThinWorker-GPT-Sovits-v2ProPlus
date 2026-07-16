"""FastAPI streaming TTS server for GPT-SoVITS v2ProPlus."""
import json
import os
import struct
import sys
import threading
import time
from io import BytesIO
from typing import Literal, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from thin_tts import __version__
from thin_tts.config import ServerConfig
from thin_tts.events import startup_events
from thin_tts.g2pw_backend import (
    configure_g2pw_backend,
    cuda_device_id_from_device,
    g2pw_backend_status,
)
from thin_tts.voice_profile import (
    build_voice_profile_metadata,
    load_voice_profile,
    save_voice_profile,
)

PIPELINE = None
_config: Optional[ServerConfig] = None
_INFERENCE_LOCK = threading.Lock()
_VOICE_PROFILE_STATUS = {
    "configured": False,
    "state": "disabled",
    "reason": None,
    "path": None,
    "voice_encoders_loaded": True,
}


def _voice_profile_metadata(cfg: ServerConfig) -> dict:
    return build_voice_profile_metadata(
        artifacts={
            "t2s": cfg.t2s_weights,
            "vits": cfg.vits_lora or cfg.vits_weights,
            "bert": cfg.bert_path,
            "hubert": cfg.hubert_path,
            "sv": cfg.sv_path,
        },
        reference_audio=cfg.ref_audio,
        prompt_text=cfg.ref_text,
        prompt_lang="zh",
    )


def _backend_status(pipeline) -> dict:
    if pipeline is None:
        return {
            "requested": None,
            "active": "unloaded",
            "state": "unloaded",
            "fallback_reason": None,
            "capture_count": 0,
            "graph_memory_mb": None,
        }
    backend = getattr(pipeline, "t2s_backend", None)
    if backend is None or not hasattr(backend, "status_dict"):
        return {
            "requested": None,
            "active": "sdpa",
            "state": "unmanaged",
            "fallback_reason": None,
            "capture_count": 0,
            "graph_memory_mb": None,
        }
    return backend.status_dict()


def _configuration_status() -> dict:
    if _config is None:
        return {
            "preset": None,
            "device": None,
            "half": None,
            "fallback_policy": None,
            "rng_isolation": None,
            "cache_vits_encoded_text": None,
        }
    return {
        "preset": _config.preset,
        "device": _config.device,
        "half": _config.half,
        "fallback_policy": _config.fallback_policy,
        "rng_isolation": _config.rng_isolation,
        "cache_vits_encoded_text": _config.cache_vits_encoded_text,
    }


def enforce_fallback_policy(
    cfg,
    *,
    t2s_status: dict,
    g2pw_status: dict,
    event_sink=startup_events.emit,
) -> None:
    """Make backend degradation visible and optionally fatal."""
    reasons = []
    if cfg.t2s_backend == "triton" and t2s_status.get("active") != "triton":
        reason = f"T2S requested triton but active backend is {t2s_status.get('active')}"
        if t2s_status.get("fallback_reason"):
            reason += f": {t2s_status['fallback_reason']}"
        reasons.append(reason)
    if cfg.g2pw_backend == "cuda" and g2pw_status.get("active") != "cuda":
        reason = f"G2PW requested cuda but active backend is {g2pw_status.get('active')}"
        if g2pw_status.get("fallback_reason"):
            reason += f": {g2pw_status['fallback_reason']}"
        reasons.append(reason)
    if not reasons:
        return
    event_sink("thin_tts_degraded", policy=cfg.fallback_policy, reasons=reasons)
    if cfg.fallback_policy == "fail":
        raise RuntimeError("; ".join(reasons))


class StreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    mode: Literal[2, 3, 4] = 4
    seed: int = 8110
    min_chunk_length: Optional[int] = Field(default=None, ge=1)
    hybrid_switch_tokens: Optional[int] = Field(default=None, ge=0)
    hybrid_steady_tokens: Optional[int] = Field(default=None, ge=0)
    hybrid_buffer_target_ms: Optional[int] = Field(default=None, ge=0)
    profile_request_id: Optional[str] = None
    rng_isolation: Optional[bool] = None
    cache_vits_encoded_text: Optional[bool] = None


def _load_pipeline():
    global PIPELINE, _VOICE_PROFILE_STATUS
    if PIPELINE is not None:
        return PIPELINE

    cfg = _config
    voice_profile_cache = None
    voice_profile_metadata = None
    with startup_events.stage("voice_profile"):
        if cfg.voice_profile:
            voice_profile_metadata = _voice_profile_metadata(cfg)
            voice_profile_cache, reason = load_voice_profile(
                cfg.voice_profile,
                expected_metadata=voice_profile_metadata,
            )
            _VOICE_PROFILE_STATUS = {
                "configured": True,
                "state": "loaded" if voice_profile_cache is not None else "rebuilding",
                "reason": reason,
                "path": str(cfg.voice_profile),
                "voice_encoders_loaded": voice_profile_cache is None,
            }
            print(
                json.dumps(
                    {"event": "thin_tts_voice_profile", **_VOICE_PROFILE_STATUS},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    # Set env vars BEFORE pipeline import — chinese2.py reads bert_path at module load time
    if cfg.bert_path:
        os.environ["bert_path"] = cfg.bert_path
    if cfg.sv_path:
        os.environ["THIN_TTS_SV_PATH"] = cfg.sv_path
    # tqdm writes synchronously on every decode step. Keep production serving
    # quiet by default while allowing an explicit false value to opt back in.
    os.environ.setdefault("THIN_TTS_DISABLE_TQDM", "1")

    with startup_events.stage("runtime_setup"):
        configure_g2pw_backend(
            requested=cfg.g2pw_backend,
            cuda_device_id=cuda_device_id_from_device(cfg.device),
            cuda_memory_limit_mb=cfg.g2pw_cuda_memory_limit_mb,
        )

    import torch
    from thin_tts.pipeline.tts import TTS, TTS_Config

    pipeline_config = {
        "custom": {
            "device": cfg.device,
            "is_half": cfg.half,
            "version": "v2ProPlus",
            "t2s_weights_path": cfg.t2s_weights,
            "vits_weights_path": cfg.vits_lora if cfg.vits_lora else cfg.vits_weights,
            "bert_base_path": cfg.bert_path,
            "cnhuhbert_base_path": cfg.hubert_path,
            "sv_path": cfg.sv_path,
            "t2s_backend": cfg.t2s_backend,
        }
    }

    pipeline_config["custom"]["t2s_backend_strict"] = cfg.fallback_policy == "fail"
    with startup_events.stage("pipeline_load"):
        tts_config = TTS_Config(pipeline_config)
        PIPELINE = TTS(tts_config, voice_profile_cache=voice_profile_cache)

    # Warmup
    print(json.dumps({"event": "thin_tts_warmup"}), flush=True)
    warmup_req = {
        "text": "你好。",
        "text_lang": "zh",
        "ref_audio_path": cfg.ref_audio,
        "prompt_text": cfg.ref_text,
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
        "cache_vits_encoded_text": True,
        "fragment_interval": 0.0,
    }
    with startup_events.stage("warmup"):
        for _ in PIPELINE.run(warmup_req):
            pass

    if cfg.voice_profile and voice_profile_cache is None:
        compiled_cache = PIPELINE.export_voice_profile_cache()
        save_voice_profile(
            cfg.voice_profile,
            metadata=voice_profile_metadata,
            cache=compiled_cache,
        )
        PIPELINE.unload_voice_encoders()
        _VOICE_PROFILE_STATUS = {
            "configured": True,
            "state": "compiled",
            "reason": None,
            "path": str(cfg.voice_profile),
            "voice_encoders_loaded": False,
        }
        print(
            json.dumps(
                {"event": "thin_tts_voice_profile", **_VOICE_PROFILE_STATUS},
                ensure_ascii=False,
            ),
            flush=True,
        )

    enforce_fallback_policy(
        cfg,
        t2s_status=_backend_status(PIPELINE),
        g2pw_status=g2pw_backend_status(),
    )

    return PIPELINE


def _make_wav_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0x7FFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0x7FFFFFFF,
    )
    return header


def _audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    return audio.tobytes()


def _request_for(
    text: str,
    cfg: ServerConfig,
    mode: int,
    seed: int,
    min_chunk_length: Optional[int] = None,
    hybrid_switch_tokens: Optional[int] = None,
    hybrid_steady_tokens: Optional[int] = None,
    hybrid_buffer_target_ms: Optional[int] = None,
    profile_request_id: Optional[str] = None,
    rng_isolation: bool = True,
    cache_vits_encoded_text: bool = True,
) -> dict:
    chunk_length = min_chunk_length if min_chunk_length is not None else 10
    # mode 4 = hybrid: early chunks use mode-2 mute-boundary detection for clean
    # splices, but if a chunk accumulates past this deadline without finding a
    # boundary, force a fixed-length cut (mode 3). Caps mode-2's worst-case
    # first-packet latency while keeping clean cuts where pauses are easy.
    #
    # Default 50 (not 2*chunk_length): below ~40-50 tokens the streaming splice
    # (SOLA overlap_frames handoff) leaks an audible repeat/rewind artifact.
    # Measured clean at chunk=50 (first-packet ~665ms, ~2x faster than mode 2).
    if mode == 4 and hybrid_switch_tokens is None:
        hybrid_switch_tokens = 50
    if mode == 4:
        if hybrid_buffer_target_ms is None:
            hybrid_buffer_target_ms = 500 if hybrid_steady_tokens is None else 0
        hybrid_buffer_target_ms = int(hybrid_buffer_target_ms)
        if hybrid_buffer_target_ms < 0:
            raise ValueError("hybrid_buffer_target_ms must not be negative")
    else:
        hybrid_buffer_target_ms = 0
    request = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": cfg.ref_audio,
        "prompt_text": cfg.ref_text,
        "prompt_lang": "zh",
        "text_split_method": "cut2",
        "batch_size": 1,
        "speed_factor": 1.0,
        "streaming_mode": mode >= 2,
        "parallel_infer": mode < 2,
        "split_bucket": mode < 2,
        "fixed_length_chunk": mode == 3,
        "top_k": 15,
        "top_p": 0.6,
        "temperature": 0.6,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "overlap_length": 2,
        "min_chunk_length": chunk_length,
        "hybrid_switch_tokens": hybrid_switch_tokens or 0,
        "hybrid_steady_tokens": hybrid_steady_tokens or 0,
        "hybrid_buffer_target_ms": hybrid_buffer_target_ms,
        "fragment_interval": 0.0,
        "seed": seed,
        "profile_request_id": profile_request_id,
        "rng_isolation": rng_isolation,
        "cache_vits_encoded_text": cache_vits_encoded_text,
    }
    return request


def _stream_audio(pipeline, request_dict: dict, sample_rate: int):
    """Stream one request while exclusively owning the stateful pipeline.

    T2S KV caches, RNG state, prompt caches, and the stop flag live on the
    singleton pipeline. Holding the lock for the generator's full lifetime
    prevents concurrent requests from corrupting that shared state. The
    context manager also releases the lock when a client disconnects and the
    response generator is closed.
    """
    with _INFERENCE_LOCK:
        yield _make_wav_header(sample_rate)
        t0 = time.perf_counter()
        chunk_idx = 0
        for _sr, audio_chunk in pipeline.run(request_dict):
            pcm_bytes = _audio_to_pcm16_bytes(audio_chunk)
            elapsed = round((time.perf_counter() - t0) * 1000)
            print(json.dumps({
                "event": "thin_tts_chunk",
                "chunk": chunk_idx,
                "elapsed_ms": elapsed,
                "bytes": len(pcm_bytes),
            }), flush=True)
            yield pcm_bytes
            chunk_idx += 1


app = FastAPI(title="Thin TTS Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "server": "thin-tts-server",
        "version": __version__,
        "loaded": PIPELINE is not None,
        "streaming": True,
        "t2s_backend": _backend_status(PIPELINE),
        "g2pw_backend": g2pw_backend_status(),
        "voice_profile": _VOICE_PROFILE_STATUS,
        "configuration": _configuration_status(),
    }


@app.post("/stream")
async def stream(req: StreamRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    cfg = _config
    pipeline = _load_pipeline()
    request_dict = _request_for(
        text=req.text,
        cfg=cfg,
        mode=req.mode,
        seed=req.seed,
        min_chunk_length=req.min_chunk_length,
        hybrid_switch_tokens=req.hybrid_switch_tokens,
        hybrid_steady_tokens=req.hybrid_steady_tokens,
        hybrid_buffer_target_ms=req.hybrid_buffer_target_ms,
        profile_request_id=req.profile_request_id,
        rng_isolation=cfg.rng_isolation if req.rng_isolation is None else req.rng_isolation,
        cache_vits_encoded_text=(
            cfg.cache_vits_encoded_text
            if req.cache_vits_encoded_text is None
            else req.cache_vits_encoded_text
        ),
    )

    sample_rate = 32000

    return StreamingResponse(
        _stream_audio(pipeline, request_dict, sample_rate),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


def run(cfg: ServerConfig):
    global _config
    _config = cfg
    print(json.dumps({"event": "thin_tts_loading"}), flush=True)
    with startup_events.stage("application_startup"):
        _load_pipeline()
    print(json.dumps({
        "event": "thin_tts_ready",
        "host": cfg.host,
        "port": cfg.port,
        "t2s_backend": _backend_status(PIPELINE),
        "g2pw_backend": g2pw_backend_status(),
        "voice_profile": _VOICE_PROFILE_STATUS,
        "configuration": _configuration_status(),
    }), flush=True)
    uvicorn.run(app, host=cfg.host, port=cfg.port)
