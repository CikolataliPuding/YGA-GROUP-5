# scripts/06_latency_check.py

import json
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score, f1_score


# ── CONFIG ──────────────────────────────────────────────────────────────────
ONNX_INT8  = "onnx/xlmr-erlikgate-int8"
ONNX_FP32  = "onnx/xlmr-erlikgate-fp32"
TEST_PATH  = "data/processed/test.jsonl"
MAX_LENGTH = 64
WARMUP     = 50
NFR_LIMIT  = 10.0
ID2LABEL   = {0: "GUVENLI", 1: "KAYTARMA", 2: "TEHDIT"}
# ────────────────────────────────────────────────────────────────────────────


def load_test(path: str):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(int(obj["label"]))
    return texts, labels


def evaluate(model_dir: str, texts: list, labels: list, model_tag: str, file_name: str = "model.onnx", max_length: int = 64):
    print(f"\n{'='*55}")
    print(f"Model: {model_tag}")
    print(f"{'='*55}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    model     = ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        file_name=file_name,
        session_options=sess_options,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        fix_mistral_regex=True,
    )

    preds     = []
    latencies = []

    for i, text in enumerate(texts):
        inp = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        t0  = time.perf_counter()
        out = model(**inp)
        ms  = (time.perf_counter() - t0) * 1000

        pred = int(np.argmax(out.logits))
        preds.append(pred)

        if i >= WARMUP:
            latencies.append(ms)

    # Doğruluk raporu — sadece ilk 45 örnek (orijinal test seti)
    print(classification_report(
        labels[:45], preds[:45],
        target_names=["GUVENLI", "KAYTARMA", "TEHDIT"],
        digits=4,
    ))

    arr = np.array(latencies)
    p50_ms = None
    p95_ms = None

    if arr.size == 0:
        print(
            f"Latency stats hesaplanamadı: ilk {WARMUP} warmup sonrası örnek kalmadı "
            f"(toplam metin sayısı: {len(texts)})."
        )
    else:
        p50_ms = float(np.percentile(arr, 50))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(np.percentile(arr, 99))
        max_ms = float(arr.max())
        avg_ms = float(arr.mean())
        nfr = p95_ms <= NFR_LIMIT

        print(f"Latency ({len(arr)} örnek, ilk {WARMUP} warmup hariç):")
        print(f"  avg_ms : {avg_ms:.3f}")
        print(f"  p50_ms : {p50_ms:.3f}")
        print(f"  p95_ms : {p95_ms:.3f}")
        print(f"  p99_ms : {p99_ms:.3f}")
        print(f"  max_ms : {max_ms:.3f}")
        print(f"  NFR-01 (p95 ≤ {NFR_LIMIT}ms): {'PASS ✓' if nfr else 'FAIL ✗'}")

    return {
        'max_len':  max_length,
        'accuracy': accuracy_score(labels[:45], preds[:45]),
        'f1_macro': f1_score(labels[:45], preds[:45], average='macro'),
        'p50_ms':   p50_ms,
        'p95_ms':   p95_ms,
    }


def main():
    print("ErlikGate — Model x Token Ablasyon Testi")
    texts, labels = load_test(TEST_PATH)
    texts_bench  = texts * 10
    labels_bench = labels * 10
    print(f"Test seti: {len(texts)} | Benchmark: {len(texts_bench)} örnek\n")

    MODELS = [
        ("xlmr",     "onnx/xlmr-int8-tok{tok}",     "model_quantized.onnx"),
        ("deberta",  "onnx/deberta-int8-tok{tok}",   "model_quantized.onnx"),
        ("minilm",   "onnx/minilm-int8-tok{tok}",    "model_quantized.onnx"),
        ("tinybert", "onnx/tinybert-int8-tok{tok}",  "model_quantized.onnx"),
        ("t5",       "onnx/t5-int8-tok{tok}",        "model_quantized.onnx"),
    ]
    TOKEN_LENGTHS = [32, 64, 128, 256]

    summary = []
    for model_name, path_template, file_name in MODELS:
        for tok in TOKEN_LENGTHS:
            model_dir = path_template.format(tok=tok)
            if not Path(model_dir).exists():
                print(f"[SKIP] {model_dir} bulunamadi")
                continue
            result = evaluate(
                model_dir, texts_bench, labels_bench,
                f"{model_name} INT8 tok={tok}",
                file_name=file_name,
                max_length=tok,
            )
            result["model"] = model_name
            summary.append(result)

    print("\n" + "="*75)
    print("OZET TABLO -- Model x Token Ablasyonu (INT8)")
    print("="*75)
    print(f"{'Model':<12} {'tok':>4} {'accuracy':>10} {'f1_macro':>10} {'p50_ms':>8} {'p95_ms':>8} {'NFR-01':>8}")
    print("-"*75)
    for r in summary:
        nfr = "PASS" if r['p95_ms'] <= 10.0 else "FAIL"
        print(f"{r['model']:<12} {r['max_len']:>4} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['p50_ms']:>8.3f} {r['p95_ms']:>8.3f} {nfr:>8}")


if __name__ == "__main__":
    main()