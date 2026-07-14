"""Compare legacy and candidate Chinese frontend segmentation paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from thin_tts.pipeline.text_preprocessor import (
    _legacy_language_segments,
    expand_word_features,
    segment_text_for_chinese_frontend,
)
from thin_tts.text import chinese2, cleaned_text_to_sequence
from thin_tts.text.cleaner import clean_text


CORPUS = {
    "plain": "今天天气很好，我们一起出去走走吧。",
    "date": "今天是2026年7月14日，星期二。",
    "numbers": "温度是零下3.5度，成功率达到99.9%。",
    "amount": "这件商品原价￥1299，现在只要￥999.50。",
    "quantifier": "请重新量一遍，这一批共有两百个样本。",
    "polyphones": "银行行长重新丈量了重庆路的长度。",
    "punctuation": "嗯？真的……可以吗？！当然可以。",
    "erhua": "胡同儿里的小院儿很有老北京范儿。",
    "mixed_ascii": "AI模型GPT-SoVITS v2ProPlus发布。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--g2pw-model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/frontend_v2/results.json"),
    )
    return parser.parse_args()


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def extract_frontend(
    segments: list[dict[str, str]],
    tokenizer,
    bert_model,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    all_phones = []
    all_phone_ids = []
    all_word2ph = []
    all_norm_text = []
    all_pinyin = []
    all_features = []

    synchronize(device)
    started = time.perf_counter()
    for segment in segments:
        phones, word2ph, norm_text = clean_text(segment["text"], "zh", "v2")
        phone_ids = cleaned_text_to_sequence(phones, "v2")
        pinyin = chinese2._get_g2pw()._g2pw([norm_text])[0] if norm_text else []

        inputs = tokenizer(norm_text, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            hidden = bert_model(**inputs, output_hidden_states=True)["hidden_states"][-3]
            word_features = hidden[0, 1:-1]
            phone_features = expand_word_features(word_features, word2ph).T

        all_phones.extend(phones)
        all_phone_ids.extend(phone_ids)
        all_word2ph.extend(word2ph)
        all_norm_text.append(norm_text)
        all_pinyin.extend(pinyin)
        all_features.append(phone_features)

    features = torch.cat(all_features, dim=1)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    record = {
        "segments": segments,
        "norm_text": "".join(all_norm_text),
        "pinyin": all_pinyin,
        "phones": all_phones,
        "phone_ids": all_phone_ids,
        "word2ph": all_word2ph,
        "bert_shape": list(features.shape),
        "bert_sha256": tensor_hash(features),
        "elapsed_ms": round(elapsed_ms, 3),
    }
    return record, features


def compare_records(left: dict, left_features: torch.Tensor, right: dict, right_features: torch.Tensor) -> dict:
    fields = ["norm_text", "pinyin", "phones", "phone_ids", "word2ph", "bert_shape", "bert_sha256"]
    field_equal = {field: left[field] == right[field] for field in fields}
    feature_equal = torch.equal(left_features, right_features)
    max_abs_diff = 0.0
    if left_features.shape == right_features.shape and not feature_equal:
        max_abs_diff = float((left_features.float() - right_features.float()).abs().max().item())
    return {
        "equal": all(field_equal.values()) and feature_equal,
        "field_equal": field_equal,
        "bert_tensor_equal": feature_equal,
        "bert_max_abs_diff": max_abs_diff,
    }


def main() -> None:
    args = parse_args()
    os.environ["bert_path"] = args.bert_path
    os.environ["g2pw_model_path"] = args.g2pw_model_path
    os.environ["THIN_TTS_G2PW_BACKEND"] = "cpu"

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(args.bert_path).eval().to(device)
    if args.half and device.type != "cpu":
        bert_model = bert_model.half()

    report = {
        "device": str(device),
        "dtype": str(next(bert_model.parameters()).dtype),
        "cases": {},
    }
    cache = {}

    def run_segments(segments):
        key = json.dumps(segments, ensure_ascii=False, sort_keys=True)
        if key not in cache:
            cache[key] = extract_frontend(segments, tokenizer, bert_model, device)
        return cache[key]

    for name, text in CORPUS.items():
        legacy_started = time.perf_counter()
        legacy_segments = _legacy_language_segments(text)
        legacy_segment_ms = (time.perf_counter() - legacy_started) * 1000

        active_started = time.perf_counter()
        active_segments = segment_text_for_chinese_frontend(text)
        active_segment_ms = (time.perf_counter() - active_started) * 1000
        whole_segments = [{"lang": "zh", "text": text}]

        legacy_record, legacy_features = run_segments(legacy_segments)
        active_record, active_features = run_segments(active_segments)
        whole_record, whole_features = run_segments(whole_segments)

        report["cases"][name] = {
            "text": text,
            "legacy_segment_ms": round(legacy_segment_ms, 3),
            "active_segment_ms": round(active_segment_ms, 3),
            "legacy": legacy_record,
            "active": active_record,
            "whole": whole_record,
            "active_vs_legacy": compare_records(
                active_record,
                active_features,
                legacy_record,
                legacy_features,
            ),
            "whole_vs_legacy": compare_records(
                whole_record,
                whole_features,
                legacy_record,
                legacy_features,
            ),
        }
        print(
            f"{name}: active={report['cases'][name]['active_vs_legacy']['equal']} "
            f"whole={report['cases'][name]['whole_vs_legacy']['equal']} "
            f"segments={len(legacy_segments)}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
