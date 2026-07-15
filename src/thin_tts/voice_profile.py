"""Versioned fixed-voice profile metadata and persistence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch


VOICE_PROFILE_SCHEMA_VERSION = 1
REQUIRED_CACHE_KEYS = {
    "ref_audio_path",
    "prompt_semantic",
    "refer_spec",
    "prompt_text",
    "prompt_lang",
    "phones",
    "bert_features",
    "norm_text",
    "aux_ref_audio_paths",
    "sv_emb_list",
}


def _artifact_signature(path_value: str) -> dict:
    path = Path(path_value).expanduser().resolve()
    stat = path.stat()
    signature = {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if path.is_dir():
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            child_stat = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "kind": "directory" if child.is_dir() else "file",
                    "size": child_stat.st_size,
                    "mtime_ns": child_stat.st_mtime_ns,
                }
            )
        signature["entries"] = entries
    return signature


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_voice_profile_metadata(
    *,
    artifacts: dict[str, str],
    reference_audio: str,
    prompt_text: str,
    prompt_lang: str,
) -> dict:
    reference_path = Path(reference_audio).expanduser().resolve()
    reference_signature = _artifact_signature(str(reference_path))
    reference_signature["sha256"] = _sha256(reference_path)
    return {
        "schema_version": VOICE_PROFILE_SCHEMA_VERSION,
        "artifacts": {
            name: _artifact_signature(path)
            for name, path in sorted(artifacts.items())
        },
        "reference_audio": reference_signature,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
    }


def save_voice_profile(path_value, *, metadata: dict, cache: dict) -> None:
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save({"metadata": metadata, "cache": cache}, temporary)
    os.replace(temporary, path)


def load_voice_profile(path_value, *, expected_metadata: dict):
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        return None, "missing"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None, "unreadable"
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        return None, "fingerprint_mismatch"
    cache = payload.get("cache")
    if not isinstance(cache, dict) or not REQUIRED_CACHE_KEYS.issubset(cache):
        return None, "invalid_cache"
    return cache, "loaded"
