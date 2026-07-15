"""Generate the final short/medium/long by mode 2/3/4 listening matrix."""

from __future__ import annotations

import argparse
import html
import json
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
MODES = (2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9881/stream")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output/final_matrix"),
    )
    return parser.parse_args()


def payload(text: str, mode: int, seed: int, request_id: str) -> dict:
    return {
        "text": text,
        "mode": mode,
        "seed": seed,
        "min_chunk_length": 10,
        "rng_isolation": True,
        "cache_vits_encoded_text": True,
        "profile_request_id": request_id,
    }


def build_html(rows: list[dict]) -> str:
    sections = []
    for text_name, text in TEXTS.items():
        cards = []
        for mode in MODES:
            row = next(
                item for item in rows if item["text"] == text_name and item["mode"] == mode
            )
            cards.append(
                "<article>"
                f"<h3>Mode {mode}</h3>"
                f"<audio controls preload='metadata' src='audio/{html.escape(row['filename'])}'></audio>"
                f"<p>TTFB {row['audio_ttfb_ms']:.1f} ms · total {row['total_ms']:.1f} ms · "
                f"audio {row['audio_s']:.2f} s</p>"
                "</article>"
            )
        sections.append(
            f"<section><h2>{html.escape(text_name)}</h2><p>{html.escape(text)}</p>"
            f"<div>{''.join(cards)}</div></section>"
        )
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Final TTS matrix</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:28px auto;padding:0 16px}"
        "section{margin:30px 0}section>div{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}"
        "article{border:1px solid #bbb;padding:14px;border-radius:6px}audio{width:100%}"
        "@media(max-width:800px){section>div{grid-template-columns:1fr}}</style>"
        f"<body><h1>Final: short / medium / long × mode 2 / 3 / 4</h1>{''.join(sections)}</body></html>"
    )


def main() -> None:
    args = parse_args()
    for text_name, text in TEXTS.items():
        for mode in MODES:
            request_id = f"warmup-final-{text_name}-mode{mode}"
            synthesize(
                args.url,
                payload(text, mode, args.seed, request_id),
                args.output_dir / "warmup" / f"{request_id}.wav",
            )

    rows = []
    for text_name, text in TEXTS.items():
        for mode in MODES:
            request_id = f"final-{text_name}-mode{mode}"
            filename = f"{request_id}.wav"
            row = synthesize(
                args.url,
                payload(text, mode, args.seed, request_id),
                args.output_dir / "audio" / filename,
            )
            row.update({"text": text_name, "mode": mode, "filename": filename})
            rows.append(row)
            print(
                f"{request_id}: ttfb={row['audio_ttfb_ms']:.1f}ms "
                f"total={row['total_ms']:.1f}ms audio={row['audio_s']:.2f}s",
                flush=True,
            )

    report = {"seed": args.seed, "texts": TEXTS, "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(build_html(rows), encoding="utf-8")
    print((args.output_dir / "index.html").resolve())


if __name__ == "__main__":
    main()
