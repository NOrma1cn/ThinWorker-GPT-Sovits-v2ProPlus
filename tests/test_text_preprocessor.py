import torch

from thin_tts.pipeline.text_preprocessor import expand_word_features


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
