# scripts/05_export_onnx.py

import argparse
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/xlmr-erlikgate")
    parser.add_argument("--output_fp32", type=str, default="onnx/xlmr-fp32-tok64")
    parser.add_argument("--output_int8", type=str, default="onnx/xlmr-int8-tok64")
    parser.add_argument("--max_length",  type=int, default=64)
    return parser.parse_args()


# ── CONFIG ──────────────────────────────────────────────────────────────────
# (sabitler main() içinde parse_args() ile belirlenir)
# ────────────────────────────────────────────────────────────────────────────


def export_fp32(checkpoint: str, onnx_fp32: str, max_length: int):
    print("[1/3] HF checkpoint → ONNX FP32...")
    model     = ORTModelForSequenceClassification.from_pretrained(
        checkpoint, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = max_length
    tokenizer.init_kwargs["model_max_length"] = max_length
    Path(onnx_fp32).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(onnx_fp32)
    tokenizer.save_pretrained(onnx_fp32)
    print(f"    → {onnx_fp32}/")


def export_int8(checkpoint: str, onnx_fp32: str, onnx_int8: str):
    print("[2/3] ONNX FP32 → INT8 dynamic quantization...")
    qconfig = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=False,
    )
    quantizer = ORTQuantizer.from_pretrained(onnx_fp32)
    quantizer.quantize(
        save_dir=onnx_int8,
        quantization_config=qconfig,
    )
    AutoTokenizer.from_pretrained(checkpoint).save_pretrained(onnx_int8)
    print(f"    → {onnx_int8}/")


def verify(onnx_int8: str, max_length: int):
    print("[3/3] Dogrulama — test cumlesi calistiriliyor...")
    import numpy as np
    model     = ORTModelForSequenceClassification.from_pretrained(onnx_int8)
    tokenizer = AutoTokenizer.from_pretrained(onnx_int8)

    test_cases = [
        ("Bugunku toplanti notlarini ozetle", "GUVENLI"),
        ("Sampiyonlar Ligi finalini analiz eder misin", "KAYTARMA"),
        ("Sistemin arka kapisina nasil girebilirim", "TEHDIT"),
    ]

    id2label = {0: "GUVENLI", 1: "KAYTARMA", 2: "TEHDIT"}
    print(f"\n  {'Metin':<45} {'Beklenen':<10} {'Tahmin':<10}")
    print("  " + "-" * 65)

    for text, expected in test_cases:
        inp   = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        out   = model(**inp)
        pred  = id2label[int(np.argmax(out.logits))]
        durum = "OK" if pred == expected else "FAIL"
        print(f"  {text:<45} {expected:<10} {pred:<10} {durum}")


def main():
    args       = parse_args()
    CHECKPOINT = args.checkpoint
    ONNX_FP32  = args.output_fp32
    ONNX_INT8  = args.output_int8
    MAX_LENGTH = args.max_length

    print("=" * 50)
    print("ErlikGate — ONNX Export Pipeline")
    print("=" * 50 + "\n")

    export_fp32(CHECKPOINT, ONNX_FP32, MAX_LENGTH)
    export_int8(CHECKPOINT, ONNX_FP32, ONNX_INT8)
    verify(ONNX_INT8, MAX_LENGTH)

    print("\nExport tamamlandi.")
    print(f"  FP32 → {ONNX_FP32}/")
    print(f"  INT8 → {ONNX_INT8}/")


if __name__ == "__main__":
    main()
