from types import SimpleNamespace

import torch

from thin_tts.models.vits import TextEncoder
from thin_tts.server import _request_for


class TextEncoderStub:
    def __init__(self):
        self.calls = 0

    def encode_target_text(self, text, text_lengths, dtype):
        self.calls += 1
        return text.to(dtype=dtype), text_lengths.to(dtype=dtype)


def test_target_text_cache_skips_encoding():
    stub = TextEncoderStub()
    cached_text = torch.ones(1, 2, 3)
    cached_mask = torch.ones(1, 1, 3)

    encoded_text, text_mask = TextEncoder.get_target_text_conditioning(
        stub,
        torch.zeros(1, 3, dtype=torch.long),
        torch.tensor([3]),
        torch.float32,
        cached_text=cached_text,
        cached_mask=cached_mask,
    )

    assert encoded_text is cached_text
    assert text_mask is cached_mask
    assert stub.calls == 0


def test_target_text_conditioning_encodes_on_cache_miss():
    stub = TextEncoderStub()
    text = torch.tensor([[1, 2, 3]])
    text_lengths = torch.tensor([3])

    encoded_text, text_mask = TextEncoder.get_target_text_conditioning(
        stub,
        text,
        text_lengths,
        torch.float32,
    )

    assert torch.equal(encoded_text, text.to(dtype=torch.float32))
    assert torch.equal(text_mask, text_lengths.to(dtype=torch.float32))
    assert stub.calls == 1


def test_server_enables_encoded_text_cache_by_default():
    cfg = SimpleNamespace(ref_audio="reference.wav", ref_text="reference text")

    request = _request_for(
        "测试文本。",
        cfg,
        mode=4,
        seed=314159,
    )

    assert request["cache_vits_encoded_text"] is True


def test_server_can_disable_encoded_text_cache():
    cfg = SimpleNamespace(ref_audio="reference.wav", ref_text="reference text")

    request = _request_for(
        "测试文本。",
        cfg,
        mode=4,
        seed=314159,
        cache_vits_encoded_text=False,
    )

    assert request["cache_vits_encoded_text"] is False
