import torch

from thin_tts.pipeline.tts import TTS


def test_final_streaming_chunk_is_crossfaded_and_keeps_its_tail():
    pipeline = object.__new__(TTS)
    previous = torch.ones(12)
    current = -torch.ones(8)

    emitted, is_first_chunk = pipeline.splice_streaming_audio_chunk(
        audio_chunk=current,
        last_audio_chunk=previous,
        overlap_size=4,
        is_first_chunk=False,
        is_final=True,
    )

    assert is_first_chunk is False
    assert emitted.shape == current.shape
    assert emitted[0].item() > 0.9
    assert emitted[-1].item() == -1.0


def test_final_streaming_chunk_can_limit_how_long_previous_audio_is_blended():
    pipeline = object.__new__(TTS)
    previous = torch.zeros(12)
    previous[-8:] = torch.tensor([1.0, 1.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0])
    current = torch.zeros(8)

    emitted, _ = pipeline.splice_streaming_audio_chunk(
        audio_chunk=current,
        last_audio_chunk=previous,
        overlap_size=8,
        fade_size=2,
        is_first_chunk=False,
        is_final=True,
    )

    assert emitted[0].item() == 1.0
    assert emitted[2:].count_nonzero().item() == 0
