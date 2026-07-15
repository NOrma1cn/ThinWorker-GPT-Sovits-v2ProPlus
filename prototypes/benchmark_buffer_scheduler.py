"""Compare legacy Mode 4 with the buffer-aware default and build listening HTML."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

from benchmark_adaptive_scheduler import synthesize


TEXTS = {
    "short": "你好呀，今天过得怎么样？",
    "medium": "嗯？你是在叫我吗？嘿嘿，有什么有趣的事情要告诉我呀？",
    "long": (
        "我们先用同一段参考音频和同一个随机种子做对比，重点听一听有没有漏字、断句、重复、爆音，"
        "或者语气和音色发生明显漂移。然后继续说一段更长的内容，确认不同切块策略不会反过来改变"
        "文本到语义模型生成的内容。"
    ),
}
POLICIES = {
    "legacy_50_50": {"hybrid_steady_tokens": 0},
    "buffer_500ms": {},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9881/stream")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output/buffer_scheduler"),
    )
    return parser.parse_args()


def payload(text: str, policy: dict, seed: int, request_id: str) -> dict:
    return {
        "text": text,
        "mode": 4,
        "seed": seed,
        "min_chunk_length": 10,
        "rng_isolation": True,
        "cache_vits_encoded_text": True,
        "profile_request_id": request_id,
        **policy,
    }


def build_html(report: dict) -> str:
    sections = []
    for text_name, text in TEXTS.items():
        cards = []
        for policy_name in POLICIES:
            summary = next(
                row
                for row in report["summaries"]
                if row["text"] == text_name and row["policy"] == policy_name
            )
            wav = Path(summary["wav"]).relative_to(report["output_dir"]).as_posix()
            cards.append(
                "<article>"
                f"<h3>{html.escape(policy_name)}</h3>"
                f"<audio controls preload='metadata' src='{html.escape(wav)}'></audio>"
                f"<p>TTFB {summary['ttfb_median_ms']:.1f} ms · "
                f"total {summary['total_median_ms']:.1f} ms</p>"
                "</article>"
            )
        sections.append(
            f"<section><h2>{html.escape(text_name)}</h2><p>{html.escape(text)}</p>"
            f"<div>{''.join(cards)}</div></section>"
        )
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Mode 4 buffer scheduler</title>"
        "<style>body{font:15px system-ui;max-width:900px;margin:28px auto;padding:0 16px}"
        "section{margin:28px 0}section>div{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
        "article{border:1px solid #bbb;padding:14px;border-radius:6px}audio{width:100%}"
        "@media(max-width:700px){section>div{grid-template-columns:1fr}}</style>"
        f"<body><h1>Mode 4: legacy 50/50 vs buffer 500 ms</h1>{''.join(sections)}</body></html>"
    )


def main() -> None:
    args = parse_args()
    rows = []
    for text_name, text in TEXTS.items():
        for policy_name, policy in POLICIES.items():
            request_id = f"buffer-warmup-{text_name}-{policy_name}"
            synthesize(
                args.url,
                payload(text, policy, args.seed, request_id),
                args.output_dir / "warmup" / f"{request_id}.wav",
            )

        for repeat in range(1, args.repeats + 1):
            policy_names = list(POLICIES)
            if repeat % 2 == 0:
                policy_names.reverse()
            for policy_name in policy_names:
                request_id = f"buffer-{text_name}-{policy_name}-r{repeat}"
                row = synthesize(
                    args.url,
                    payload(text, POLICIES[policy_name], args.seed, request_id),
                    args.output_dir / "audio" / f"{request_id}.wav",
                )
                row.update(
                    {"text": text_name, "policy": policy_name, "repeat": repeat}
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
                row
                for row in rows
                if row["text"] == text_name and row["policy"] == policy_name
            ]
            summaries.append(
                {
                    "text": text_name,
                    "policy": policy_name,
                    "ttfb_median_ms": statistics.median(
                        row["audio_ttfb_ms"] for row in selected
                    ),
                    "total_median_ms": statistics.median(
                        row["total_ms"] for row in selected
                    ),
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
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(build_html(report), encoding="utf-8")
    print((args.output_dir / "index.html").resolve())


if __name__ == "__main__":
    main()
