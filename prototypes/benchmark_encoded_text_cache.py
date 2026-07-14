"""Benchmark exact-equivalence caching of encoded VITS target text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_adaptive_scheduler import TEXTS, median, synthesize


POLICIES = {"baseline": False, "cache_encoded_text": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9881/stream")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output/encoded_text_cache"),
    )
    return parser.parse_args()


def payload_for(text: str, *, enabled: bool, seed: int, request_id: str) -> dict:
    return {
        "text": text,
        "mode": 4,
        "seed": seed,
        "min_chunk_length": 10,
        "hybrid_switch_tokens": 50,
        "hybrid_steady_tokens": 50,
        "rng_isolation": True,
        "cache_vits_encoded_text": enabled,
        "profile_request_id": request_id,
    }


def main() -> None:
    args = parse_args()
    rows = []

    for text_name, text in TEXTS.items():
        for policy_name, enabled in POLICIES.items():
            request_id = f"warmup-text-cache-{text_name}-{policy_name}"
            synthesize(
                args.url,
                payload_for(
                    text,
                    enabled=enabled,
                    seed=args.seed,
                    request_id=request_id,
                ),
                args.output_dir / "warmup" / f"{request_id}.wav",
            )

        for repeat in range(args.repeats):
            policy_names = list(POLICIES)
            if repeat % 2:
                policy_names.reverse()
            for policy_name in policy_names:
                request_id = f"text-cache-{text_name}-{policy_name}-r{repeat + 1}"
                row = synthesize(
                    args.url,
                    payload_for(
                        text,
                        enabled=POLICIES[policy_name],
                        seed=args.seed,
                        request_id=request_id,
                    ),
                    args.output_dir / "audio" / f"{request_id}.wav",
                )
                row.update(
                    {"text": text_name, "policy": policy_name, "repeat": repeat + 1}
                )
                rows.append(row)
                print(
                    f"{request_id}: ttfb={row['audio_ttfb_ms']:.1f}ms "
                    f"total={row['total_ms']:.1f}ms",
                    flush=True,
                )

    summaries = []
    parity = {}
    for text_name in TEXTS:
        selected_text = [row for row in rows if row["text"] == text_name]
        parity[text_name] = len({row["wav_sha256"] for row in selected_text}) == 1
        for policy_name in POLICIES:
            selected = [
                row for row in selected_text if row["policy"] == policy_name
            ]
            summaries.append(
                {
                    "text": text_name,
                    "policy": policy_name,
                    "audio_ttfb_median_ms": median(selected, "audio_ttfb_ms"),
                    "total_median_ms": median(selected, "total_ms"),
                    "wav_sha256": selected[0]["wav_sha256"],
                    "deterministic": len({row["wav_sha256"] for row in selected}) == 1,
                }
            )

    report = {
        "seed": args.seed,
        "repeats": args.repeats,
        "pcm_bitwise_equal": parity,
        "summaries": summaries,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"pcm_bitwise_equal": parity, "results": str(result_path.resolve())}))


if __name__ == "__main__":
    main()
