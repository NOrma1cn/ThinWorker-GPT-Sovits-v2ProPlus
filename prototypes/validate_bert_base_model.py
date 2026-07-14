"""Validate hidden-state parity and cost of skipping the RoBERTa MLM head."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


TEXTS = {
    "opening": "嗯？你是在叫我吗？嘿嘿，有什么有趣的事情要告诉我呀？",
    "medium": "如果我们只改推理实现，不改变采样参数，声音质量应该怎么判断有没有退化？",
    "long": "我们先用同一段参考音频和同一个随机种子做对比，重点听一听有没有漏字、断句、重复、爆音，或者语气和音色发生明显漂移。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/bert_base_model/results.json"),
    )
    return parser.parse_args()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_hidden(model, inputs, *, base_model: bool):
    target = model.base_model if base_model else model
    output = target(**inputs, output_hidden_states=True)
    return output["hidden_states"][-3][0, 1:-1]


def benchmark_once(model, inputs, device: torch.device, *, base_model: bool):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0

    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        hidden = forward_hidden(model, inputs, base_model=base_model)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000

    peak_delta_mib = 0.0
    if device.type == "cuda":
        peak_delta_mib = (torch.cuda.max_memory_allocated(device) - baseline) / (1024 * 1024)
    return hidden, elapsed_ms, peak_delta_mib


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    model = AutoModelForMaskedLM.from_pretrained(args.bert_path).eval().to(device)
    if args.half and device.type != "cpu":
        model = model.half()

    report = {
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "repeats": args.repeats,
        "cases": {},
    }

    for name, text in TEXTS.items():
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        benchmark_once(model, inputs, device, base_model=False)
        benchmark_once(model, inputs, device, base_model=True)

        wrapper_times = []
        wrapper_peaks = []
        base_times = []
        base_peaks = []
        wrapper_hidden = None
        base_hidden = None
        for repeat in range(args.repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for use_base_model in order:
                hidden, elapsed, peak = benchmark_once(
                    model, inputs, device, base_model=use_base_model
                )
                if use_base_model:
                    base_hidden = hidden
                    base_times.append(elapsed)
                    base_peaks.append(peak)
                else:
                    wrapper_hidden = hidden
                    wrapper_times.append(elapsed)
                    wrapper_peaks.append(peak)

        equal = torch.equal(wrapper_hidden, base_hidden)
        max_abs_diff = float((wrapper_hidden.float() - base_hidden.float()).abs().max().item())
        case = {
            "text": text,
            "tokens": int(inputs["input_ids"].shape[1]),
            "shape": list(wrapper_hidden.shape),
            "equal": equal,
            "max_abs_diff": max_abs_diff,
            "wrapper_sha256": tensor_hash(wrapper_hidden),
            "base_sha256": tensor_hash(base_hidden),
            "wrapper_median_ms": round(statistics.median(wrapper_times), 3),
            "base_median_ms": round(statistics.median(base_times), 3),
            "saved_median_ms": round(
                statistics.median(wrapper_times) - statistics.median(base_times), 3
            ),
            "wrapper_peak_delta_mib": round(statistics.median(wrapper_peaks), 3),
            "base_peak_delta_mib": round(statistics.median(base_peaks), 3),
        }
        report["cases"][name] = case
        print(
            f"{name}: equal={equal} wrapper={case['wrapper_median_ms']:.3f}ms "
            f"base={case['base_median_ms']:.3f}ms saved={case['saved_median_ms']:.3f}ms",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
