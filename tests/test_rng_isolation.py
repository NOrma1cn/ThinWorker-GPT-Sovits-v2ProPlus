import torch

from thin_tts.models.t2s_utils import multinomial_sample_one_no_sync
from thin_tts.models.vits import randn_like_with_generator
from thin_tts.pipeline.tts import create_request_generators


def test_t2s_sampling_generator_is_independent_from_global_rng():
    probabilities = torch.full((1, 32), 1 / 32)
    first_generator = torch.Generator().manual_seed(314159)
    second_generator = torch.Generator().manual_seed(314159)

    first = [
        multinomial_sample_one_no_sync(probabilities, generator=first_generator).item()
        for _ in range(8)
    ]
    torch.manual_seed(1)
    torch.rand(4096)
    second = [
        multinomial_sample_one_no_sync(probabilities, generator=second_generator).item()
        for _ in range(8)
    ]

    assert first == second


def test_vits_noise_generator_is_independent_from_global_rng():
    template = torch.empty((1, 4, 16))
    first_generator = torch.Generator().manual_seed(271828)
    second_generator = torch.Generator().manual_seed(271828)

    first = randn_like_with_generator(template, first_generator)
    torch.manual_seed(2)
    torch.rand(4096)
    second = randn_like_with_generator(template, second_generator)

    assert torch.equal(first, second)


def test_request_generators_are_repeatable_and_separate():
    first_t2s, first_vits = create_request_generators(314159, "cpu", enabled=True)
    second_t2s, second_vits = create_request_generators(314159, "cpu", enabled=True)

    first_t2s_values = torch.rand(16, generator=first_t2s)
    first_vits_values = torch.rand(16, generator=first_vits)
    second_t2s_values = torch.rand(16, generator=second_t2s)
    second_vits_values = torch.rand(16, generator=second_vits)

    assert torch.equal(first_t2s_values, second_t2s_values)
    assert torch.equal(first_vits_values, second_vits_values)
    assert not torch.equal(first_t2s_values, first_vits_values)


def test_request_generators_remain_disabled_by_default():
    assert create_request_generators(314159, "cpu", enabled=False) == (None, None)
