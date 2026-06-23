"""FastAPI streaming TTS server for GPT-SoVITS v2ProPlus."""
import json
import os
import struct
import sys
import time
from io import BytesIO
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from thin_tts.config import ServerConfig

PIPELINE = None
_config: Optional[ServerConfig] = None


class StreamRequest(BaseModel):
    text: str
    mode: int = 2
    seed: int = 8110
    min_chunk_length: Optional[int] = None
    hybrid_switch_tokens: Optional[int] = None
    profile_request_id: Optional[str] = None


def _load_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    cfg = _config

    # Set env vars BEFORE pipeline import — chinese2.py reads bert_path at module load time
    if cfg.bert_path:
        os.environ["bert_path"] = cfg.bert_path
    if cfg.sv_path:
        os.environ["THIN_TTS_SV_PATH"] = cfg.sv_path

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
        }
    }

    tts_config = TTS_Config(pipeline_config)
    PIPELINE = TTS(tts_config)

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
        "fragment_interval": 0.0,
    }
    for _ in PIPELINE.run(warmup_req):
        pass

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
    profile_request_id: Optional[str] = None,
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
        "fragment_interval": 0.0,
        "seed": seed,
        "profile_request_id": profile_request_id,
    }
    return request


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
        "loaded": PIPELINE is not None,
        "streaming": True,
    }


@app.post("/stream")
async def stream(req: StreamRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    cfg = _config
    pipeline = _load_pipeline()
    request_dict = _request_for(
        req.text,
        cfg,
        req.mode,
        req.seed,
        req.min_chunk_length,
        req.hybrid_switch_tokens,
        req.profile_request_id,
    )

    sample_rate = 32000

    def audio_generator():
        yield _make_wav_header(sample_rate)
        t0 = time.perf_counter()
        chunk_idx = 0
        for sr, audio_chunk in pipeline.run(request_dict):
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

    return StreamingResponse(
        audio_generator(),
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
    _load_pipeline()
    print(json.dumps({"event": "thin_tts_ready", "host": cfg.host, "port": cfg.port}), flush=True)
    uvicorn.run(app, host=cfg.host, port=cfg.port)
