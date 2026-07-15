from types import SimpleNamespace

import torch

from thin_tts.pipeline.tts import TTS


def populated_cache():
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


def bare_pipeline():
    pipeline = object.__new__(TTS)
    pipeline.configs = SimpleNamespace(device="cpu")
    pipeline.prompt_cache = {}
    pipeline.cnhuhbert_model = object()
    pipeline.sv_model = object()
    return pipeline


def test_export_voice_profile_moves_tensors_to_cpu():
    pipeline = bare_pipeline()
    pipeline.prompt_cache = populated_cache()

    exported = pipeline.export_voice_profile_cache()

    assert exported["prompt_semantic"].device.type == "cpu"
    assert exported["refer_spec"][0][0].device.type == "cpu"
    assert exported["sv_emb_list"][0].device.type == "cpu"


def test_restore_voice_profile_populates_prompt_cache():
    pipeline = bare_pipeline()
    cache = populated_cache()

    pipeline.restore_voice_profile_cache(cache)

    assert pipeline.prompt_cache["ref_audio_path"] == "reference.wav"
    assert torch.equal(pipeline.prompt_cache["prompt_semantic"], torch.tensor([1, 2, 3]))
    assert len(pipeline.prompt_cache["sv_emb_list"]) == 1


def test_unload_voice_encoders_removes_model_references():
    pipeline = bare_pipeline()

    pipeline.unload_voice_encoders()

    assert pipeline.cnhuhbert_model is None
    assert pipeline.sv_model is None
