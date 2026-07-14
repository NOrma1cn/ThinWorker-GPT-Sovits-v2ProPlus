"""Benchmark and build a listening set for the adaptive mode-4 deadline."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import statistics
import time
from pathlib import Path

import numpy as np
import requests


TEXTS = {
    "opening": "嗯？你是在叫我吗？嘿嘿，有什么有趣的事情要告诉我呀？",
    "long": (
        "我们先用同一段参考音频和同一个随机种子做对比，重点听一听有没有漏字、断句、重复、爆音，"
        "或者语气和音色发生明显漂移。然后继续说一段更长的内容，确认不同切块策略不会反过来改变"
        "文本到语义模型生成的内容。"
    ),
}

POLICIES = {
    "baseline_50_50": {"hybrid_switch_tokens": 50, "hybrid_steady_tokens": 50},
    "adaptive_50_120": {"hybrid_switch_tokens": 50, "hybrid_steady_tokens": 120},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9881/stream")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output/adaptive_scheduler"),
    )
    return parser.parse_args()


def audio_metrics(body: bytes) -> dict:
    pcm = np.frombuffer(body[44:], dtype="<i2").astype(np.int32)
    jumps = np.abs(np.diff(pcm)) if pcm.size > 1 else np.zeros(1, dtype=np.int32)
    return {
        "audio_s": round(pcm.size / 32000, 6),
        "max_jump": int(jumps.max(initial=0)),
        "p999_jump": float(np.percentile(jumps, 99.9)),
        "clipped_samples": int(np.count_nonzero(np.abs(pcm) >= 32767)),
    }


def synthesize(url: str, payload: dict, output: Path) -> dict:
    started = time.perf_counter()
    first_audio = None
    body = bytearray()
    with requests.post(url, json=payload, stream=True, timeout=180) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            body.extend(chunk)
            if first_audio is None and len(body) > 44:
                first_audio = time.perf_counter()
    finished = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(body)
    result = {
        "profile_request_id": payload["profile_request_id"],
        "audio_ttfb_ms": round(((first_audio or finished) - started) * 1000, 3),
        "total_ms": round((finished - started) * 1000, 3),
        "bytes": len(body),
        "wav_sha256": hashlib.sha256(body).hexdigest(),
        "wav": str(output.resolve()),
    }
    result.update(audio_metrics(body))
    return result


def median(rows: list[dict], field: str) -> float:
    return round(statistics.median(row[field] for row in rows), 3)


def build_html(report: dict) -> str:
    sections = []
    for text_name, text in TEXTS.items():
        samples = []
        for policy_name in POLICIES:
            summary = next(
                item
                for item in report["summaries"]
                if item["text"] == text_name and item["policy"] == policy_name
            )
            wav = Path(summary["wav"]).relative_to(report["output_dir"]).as_posix()
            samples.append(
                f"<article><h3>{html.escape(policy_name)}</h3>"
                f"<audio controls preload='metadata' src='{html.escape(wav)}'></audio>"
                f"<p>TTFB {summary['audio_ttfb_median_ms']:.1f} ms | "
                f"total {summary['total_median_ms']:.1f} ms | "
                f"{summary['audio_s']:.3f} s</p>"
                f"<p>deterministic: {summary['deterministic']}</p></article>"
            )
        sections.append(
            f"<section><h2>{html.escape(text_name)}</h2><p>{html.escape(text)}</p>"
            f"<div>{''.join(samples)}</div></section>"
        )
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Adaptive scheduler listening</title>"
        "<style>body{font:15px system-ui;max-width:960px;margin:32px auto;padding:0 16px}"
        "section{margin:32px 0}section>div{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
        "article{border:1px solid #bbb;padding:16px;border-radius:6px}audio{width:100%}"
        "@media(max-width:700px){section>div{grid-template-columns:1fr}}</style>"
        f"<body><h1>Mode 4: 50/50 vs 50/120</h1>{''.join(sections)}</body></html>"
    )


def main() -> None:
    args = parse_args()
    rows = []

    def payload_for(text, policy, request_id):
        return {
            "text": text,
            "mode": 4,
            "seed": args.seed,
            "min_chunk_length": 10,
            "rng_isolation": True,
            "profile_request_id": request_id,
            **policy,
        }

    for text_name, text in TEXTS.items():
        for policy_name, policy in POLICIES.items():
            warmup_id = f"warmup-{text_name}-{policy_name}"
            synthesize(
                args.url,
                payload_for(text, policy, warmup_id),
                args.output_dir / "warmup" / f"{warmup_id}.wav",
            )

        for repeat in range(args.repeats):
            policy_names = list(POLICIES)
            if repeat % 2:
                policy_names.reverse()
            for policy_name in policy_names:
                policy = POLICIES[policy_name]
                request_id = f"sched-{text_name}-{policy_name}-r{repeat + 1}"
                wav = args.output_dir / "audio" / f"{request_id}.wav"
                row = synthesize(args.url, payload_for(text, policy, request_id), wav)
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
    for text_name in TEXTS:
        for policy_name in POLICIES:
            selected = [
                row for row in rows if row["text"] == text_name and row["policy"] == policy_name
            ]
            summaries.append(
                {
                    "text": text_name,
                    "policy": policy_name,
                    "audio_ttfb_median_ms": median(selected, "audio_ttfb_ms"),
                    "total_median_ms": median(selected, "total_ms"),
                    "audio_s": selected[-1]["audio_s"],
                    "deterministic": len({row["wav_sha256"] for row in selected}) == 1,
                    "wav": selected[-1]["wav"],
                }
            )

    report = {
        "seed": args.seed,
        "output_dir": str(args.output_dir.resolve()),
        "rows": rows,
        "summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(build_html(report), encoding="utf-8")
    print((args.output_dir / "index.html").resolve())


if __name__ == "__main__":
    main()
