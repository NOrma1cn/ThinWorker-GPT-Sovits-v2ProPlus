"""Check whether right-padded VITS buckets preserve exact valid audio."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from thin_tts.models.vits import SynthesizerTrn
from thin_tts.process_ckpt import (
    get_sovits_version_from_path_fast,
    load_sovits_new,
)


BUCKETS = (32, 64, 96, 104)
LENGTHS = (22, 26, 30, 54, 70, 86, 100, 104)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("eval_config_wsl_rng.yaml"))
    return parser.parse_args()


def load_model(config_path: Path) -> SynthesizerTrn:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    weights_path = config["weights"]["vits_weights"]
    checkpoint = load_sovits_new(weights_path)
    hps = checkpoint["config"]
    hps["model"]["semantic_frame_rate"] = "25hz"
    _, model_version, _ = get_sovits_version_from_path_fast(weights_path)
    if "Pro" in model_version:
        hps["model"]["version"] = model_version

    model = SynthesizerTrn(
        hps["data"]["filter_length"] // 2 + 1,
        hps["train"]["segment_size"] // hps["data"]["hop_length"],
        n_speakers=hps["data"]["n_speakers"],
        **hps["model"],
    )
    model.load_state_dict(checkpoint["weight"], strict=False)
    return model.eval().half().cuda()


def comparison(exact: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = (exact.float() - candidate.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(exact, candidate)),
        "different_values": int(torch.count_nonzero(delta).item()),
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
    }


def audio_comparison(
    exact: torch.Tensor,
    candidate: torch.Tensor,
    *,
    protected_tail_samples: int,
) -> dict:
    exact = exact[:, :1]
    candidate = candidate[:, :1]
    result = comparison(exact, candidate)
    different_samples = torch.any(exact != candidate, dim=(0, 1))
    indices = torch.nonzero(different_samples, as_tuple=False).flatten()
    total_samples = exact.shape[-1]
    result.update(
        {
            "total_samples": total_samples,
            "first_different_sample": int(indices[0].item()) if indices.numel() else None,
            "affected_tail_samples": (
                total_samples - int(indices[0].item()) if indices.numel() else 0
            ),
            "prefix_before_protected_tail_equal": bool(
                torch.equal(
                    exact[..., :-protected_tail_samples],
                    candidate[..., :-protected_tail_samples],
                )
            ),
        }
    )
    return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model = load_model(args.config)
    torch.manual_seed(314159)
    ge = torch.randn(1, model.gin_channels, 1, device="cuda", dtype=torch.float16)
    upsample = math.prod(model.upsample_rates)
    rows = []

    for length in LENGTHS:
        bucket = next(value for value in BUCKETS if value >= length)
        z_p = torch.randn(
            1,
            model.inter_channels,
            length,
            device="cuda",
            dtype=torch.float16,
        )
        mask = torch.ones(1, 1, length, device="cuda", dtype=torch.float16)
        exact_z = model.flow(z_p, mask, g=ge, reverse=True)
        exact_audio = model.dec(exact_z * mask, g=ge)

        padding = bucket - length
        padded_z_p = F.pad(z_p, (0, padding))
        padded_mask = F.pad(mask, (0, padding))
        bucket_z = model.flow(padded_z_p, padded_mask, g=ge, reverse=True)
        flow_candidate = bucket_z[:, :, :length]
        bucket_audio = model.dec(bucket_z * padded_mask, g=ge)
        audio_candidate = bucket_audio[:, :, : length * upsample]

        row = {
            "length": length,
            "bucket": bucket,
            "flow": comparison(exact_z, flow_candidate),
            "flow_and_decoder": audio_comparison(
                exact_audio,
                audio_candidate,
                protected_tail_samples=4 * upsample,
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
