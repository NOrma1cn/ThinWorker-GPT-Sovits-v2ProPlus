from pathlib import Path

import torch

from thin_tts.voice_profile import (
    build_voice_profile_metadata,
    load_voice_profile,
    save_voice_profile,
)


def metadata_for(tmp_path: Path, *, prompt_text: str = "参考文本") -> dict:
    artifacts = {}
    for name in ("t2s", "vits", "bert", "hubert", "sv"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        artifacts[name] = str(path)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-audio")
    return build_voice_profile_metadata(
        artifacts=artifacts,
        reference_audio=str(reference),
        prompt_text=prompt_text,
        prompt_lang="zh",
    )


def cache_payload() -> dict:
    return {
        "ref_audio_path": "reference.wav",
        "prompt_semantic": torch.tensor([1, 2, 3]),
        "refer_spec": [(torch.ones(1, 2, 3), torch.ones(1, 8))],
        "prompt_text": "参考文本。",
        "prompt_lang": "zh",
        "phones": torch.tensor([1, 2]),
        "bert_features": torch.ones(4, 2),
        "norm_text": "参考文本。",
        "aux_ref_audio_paths": [],
        "sv_emb_list": [torch.ones(1, 4)],
    }


def test_voice_profile_round_trip(tmp_path):
    metadata = metadata_for(tmp_path)
    cache = cache_payload()
    profile_path = tmp_path / "voice-profile.pt"

    save_voice_profile(profile_path, metadata=metadata, cache=cache)
    loaded, reason = load_voice_profile(profile_path, expected_metadata=metadata)

    assert reason == "loaded"
    assert torch.equal(loaded["prompt_semantic"], cache["prompt_semantic"])


def test_voice_profile_rejects_prompt_change(tmp_path):
    metadata = metadata_for(tmp_path)
    profile_path = tmp_path / "voice-profile.pt"
    save_voice_profile(
        profile_path,
        metadata=metadata,
        cache=cache_payload(),
    )
    changed = dict(metadata)
    changed["prompt_text"] = "另一段参考文本"

    loaded, reason = load_voice_profile(profile_path, expected_metadata=changed)

    assert loaded is None
    assert reason == "fingerprint_mismatch"


def test_voice_profile_rejects_reference_change(tmp_path):
    metadata = metadata_for(tmp_path)
    profile_path = tmp_path / "voice-profile.pt"
    save_voice_profile(
        profile_path,
        metadata=metadata,
        cache=cache_payload(),
    )
    reference = Path(metadata["reference_audio"]["path"])
    reference.write_bytes(b"changed-reference-audio")
    changed = build_voice_profile_metadata(
        artifacts={name: item["path"] for name, item in metadata["artifacts"].items()},
        reference_audio=str(reference),
        prompt_text=metadata["prompt_text"],
        prompt_lang=metadata["prompt_lang"],
    )

    loaded, reason = load_voice_profile(profile_path, expected_metadata=changed)

    assert loaded is None
    assert reason == "fingerprint_mismatch"


def test_voice_profile_rejects_checkpoint_change(tmp_path):
    metadata = metadata_for(tmp_path)
    profile_path = tmp_path / "voice-profile.pt"
    save_voice_profile(
        profile_path,
        metadata=metadata,
        cache=cache_payload(),
    )
    t2s_path = Path(metadata["artifacts"]["t2s"]["path"])
    t2s_path.write_text("changed checkpoint", encoding="utf-8")
    changed = build_voice_profile_metadata(
        artifacts={name: item["path"] for name, item in metadata["artifacts"].items()},
        reference_audio=metadata["reference_audio"]["path"],
        prompt_text=metadata["prompt_text"],
        prompt_lang=metadata["prompt_lang"],
    )

    loaded, reason = load_voice_profile(profile_path, expected_metadata=changed)

    assert loaded is None
    assert reason == "fingerprint_mismatch"


def test_missing_voice_profile_is_reported(tmp_path):
    metadata = metadata_for(tmp_path)

    loaded, reason = load_voice_profile(
        tmp_path / "missing.pt",
        expected_metadata=metadata,
    )

    assert loaded is None
    assert reason == "missing"


def test_voice_profile_rejects_incomplete_cache(tmp_path):
    metadata = metadata_for(tmp_path)
    profile_path = tmp_path / "voice-profile.pt"
    save_voice_profile(
        profile_path,
        metadata=metadata,
        cache={"prompt_semantic": torch.tensor([1])},
    )

    loaded, reason = load_voice_profile(profile_path, expected_metadata=metadata)

    assert loaded is None
    assert reason == "invalid_cache"
