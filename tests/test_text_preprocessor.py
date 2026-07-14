import torch

from thin_tts.pipeline import text_preprocessor
from thin_tts.pipeline.text_preprocessor import (
    expand_word_features,
    segment_text_for_chinese_frontend,
)


def test_expand_word_features_preserves_device_and_order():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_features = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        device=device,
    )

    phone_features = expand_word_features(word_features, [2, 1, 3])

    assert phone_features.device == word_features.device
    assert phone_features.tolist() == [
        [1.0, 10.0],
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [3.0, 30.0],
        [3.0, 30.0],
    ]


def test_chinese_frontend_skips_language_detection_for_supported_text(monkeypatch):
    def fail_if_called(_text):
        raise AssertionError("legacy language detection should not run")

    monkeypatch.setattr(text_preprocessor, "_legacy_language_segments", fail_if_called)

    text = "2026年7月14日，重新量一遍，价格是￥3.50！"

    assert segment_text_for_chinese_frontend(text) == [{"lang": "zh", "text": text}]


def test_chinese_frontend_preserves_legacy_fallback_for_mixed_text(monkeypatch):
    calls = []

    def fake_legacy_segments(text):
        calls.append(text)
        return [{"lang": "legacy", "text": text}]

    monkeypatch.setattr(text_preprocessor, "_legacy_language_segments", fake_legacy_segments)

    mixed_texts = ["AI模型", "中文かな", "中文🙂", "扩展字𠀀"]
    for text in mixed_texts:
        assert segment_text_for_chinese_frontend(text) == [{"lang": "legacy", "text": text}]
    assert calls == mixed_texts
