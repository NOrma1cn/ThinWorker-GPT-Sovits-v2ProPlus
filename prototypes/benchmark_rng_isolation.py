"""Exercise chunk schedules with and without request-local RNG streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


TEXT = (
    "我们先用同一段参考音频和同一个随机种子做对比，重点听一听有没有漏字、断句、重复、爆音，"
    "或者语气和音色发生明显漂移。然后继续说一段更长的内容，确认不同切块策略不会反过来改变"
    "文本到语义模型生成的内容。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9881/stream")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--rng-mode",
        choices=["legacy", "isolated", "both"],
        default="both",
    )
    parser.add_argument("--chunk-lengths", type=int, nargs="+", default=[40, 80])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/rng_isolation/results.json"),
    )
    return parser.parse_args()


def synthesize(url: str, payload: dict, output: Path) -> dict:
    started = time.perf_counter()
    first_byte = None
    first_audio = None
    body = bytearray()
    with requests.post(url, json=payload, stream=True, timeout=180) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            if first_byte is None:
                first_byte = time.perf_counter()
            body.extend(chunk)
            if first_audio is None and len(body) > 44:
                first_audio = time.perf_counter()
    finished = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(body)
    return {
        "profile_request_id": payload["profile_request_id"],
        "ttfb_ms": round(((first_byte or finished) - started) * 1000, 3),
        "audio_ttfb_ms": round(((first_audio or finished) - started) * 1000, 3),
        "total_ms": round((finished - started) * 1000, 3),
        "bytes": len(body),
        "wav_sha256": hashlib.sha256(body).hexdigest(),
        "wav": str(output),
    }


def main() -> None:
    args = parse_args()
    rows = []
    audio_dir = args.output.parent / "audio"
    isolation_modes = {
        "legacy": (False,),
        "isolated": (True,),
        "both": (False, True),
    }[args.rng_mode]
    for isolated in isolation_modes:
        for chunk_length in args.chunk_lengths:
            for repeat in range(args.repeats):
                case = f"rng{int(isolated)}-chunk{chunk_length}-r{repeat + 1}"
                payload = {
                    "text": TEXT,
                    "mode": 3,
                    "seed": args.seed,
                    "min_chunk_length": chunk_length,
                    "rng_isolation": isolated,
                    "profile_request_id": case,
                }
                row = synthesize(args.url, payload, audio_dir / f"{case}.wav")
                row.update(
                    {
                        "rng_isolation": isolated,
                        "chunk_length": chunk_length,
                        "repeat": repeat + 1,
                    }
                )
                rows.append(row)
                print(
                    f"{case}: audio_ttfb={row['audio_ttfb_ms']:.1f}ms "
                    f"total={row['total_ms']:.1f}ms "
                    f"bytes={row['bytes']} hash={row['wav_sha256'][:12]}",
                    flush=True,
                )

    report = {"seed": args.seed, "text": TEXT, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
