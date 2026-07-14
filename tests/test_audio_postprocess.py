import numpy as np
import torch

from thin_tts.pipeline.tts import TTS


def _postprocess(samples):
    pipeline = object.__new__(TTS)
    _sample_rate, pcm = pipeline.audio_postprocess(
        [[torch.tensor(samples, dtype=torch.float32)]],
        32000,
        split_bucket=False,
        fragment_interval=0.0,
    )
    return pcm


def test_positive_full_scale_does_not_wrap_to_negative_pcm16():
    pcm = _postprocess([1.0])

    assert pcm.dtype == np.int16
    assert pcm.tolist() == [32767]


def test_peak_normalized_fragment_preserves_pcm16_polarity():
    pcm = _postprocess([2.0, -2.0, 1.0, -1.0])

    assert pcm.tolist() == [32767, -32767, 16383, -16383]
