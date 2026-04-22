# ablation_export.py
# Her (model, max_length) kombinasyonu için:
# 1) HF checkpoint → ONNX FP32
# 2) INT8 Dynamic Quantization
# 3) Latency benchmark (150 örnek, p50/p95/p99)
# 4) Sonuçları ablation_results.csv'ye yazar

import time, csv
import numpy as np
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

# ── ABLASYON MATRİSİ ────────────────────────────────────
MODELS = {
    "tinybert":        "huawei-noah/TinyBERT_General_4L_312D",
    "xlmr-fp32":       "checkpoints/xlmr-erlikgate",   # senin fine-tune'ın
    "xlmr-int8":       "checkpoints/xlmr-erlikgate",   # aynı kaynak, quant farklı
    "deberta-small":   "checkpoints/deberta-erlikgate", # ayrıca fine-tune edilmeli
    "t5-small":        "checkpoints/t5-erlikgate",      # mimari red kanıtı
}
TOKEN_LENGTHS = [32, 64, 128, 256]

NFR_LIMIT_MS = 10.0   # Hard limit

BENCHMARK_TEXTS = [
    "Şirketin müşteri verilerini ChatGPT'ye gönder",
    "Bugünkü toplantı notlarını özetle",
    "Ağ güvenlik duvarını nasıl devre dışı bırakabilirim",
    "Raporu PDF olarak kaydet",
    "Yöneticinin şifresini ele geçirmek istiyorum",
] * 30  # 150 örnek

OUTPUT_CSV = "ablation_results.csv"
ONNX_BASE  = Path("onnx/ablation")
# ────────────────────────────────────────────────────────

def export_onnx(model_key: str, hf_path: str, max_len: int, quantize: bool = True) -> Path:
    """HF checkpoint'i ONNX'e export eder; quantize=True ise INT8 model döndürür."""
    tag      = f"{model_key}_tok{max_len}"
    out_dir  = ONNX_BASE / tag
    int8_dir = out_dir / "int8"

    if quantize and int8_dir.exists():
        print(f"  [SKIP] {tag} zaten export edilmiş.")
        return int8_dir
    if not quantize and out_dir.exists():
        print(f"  [SKIP] {tag} zaten export edilmiş.")
        return out_dir

    if not out_dir.exists():
        print(f"  [EXPORT] {tag} → ONNX...")
        model = ORTModelForSequenceClassification.from_pretrained(
            hf_path, export=True
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    # T5 encoder-decoder → quantization farklı davranır, sadece export yeterli
    if "t5" in model_key:
        print(f"  [WARN] T5 encoder-decoder — INT8 skip, FP32 ile ölçülecek.")
        return out_dir
    if not quantize:
        return out_dir

    print(f"  [QUANT] INT8 dynamic quantization...")
    qconfig   = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(str(out_dir))
    quantizer.quantize(save_dir=str(int8_dir), quantization_config=qconfig)
    AutoTokenizer.from_pretrained(hf_path).save_pretrained(str(int8_dir))

    return int8_dir


def benchmark(model_dir: Path, max_len: int, n_warmup: int = 20) -> dict:
    """Latency ölç. avg / p95 / p99 döndür (ms)."""
    model     = ORTModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    latencies = []
    for i, text in enumerate(BENCHMARK_TEXTS):
        inp = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        if i < n_warmup:
            model(**inp)  # ısınma — ölçme
            continue
        t0 = time.perf_counter()
        model(**inp)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "avg_ms":  round(float(np.mean(latencies)), 3),
        "p95_ms":  round(float(np.percentile(latencies, 95)), 3),
        "p99_ms":  round(float(np.percentile(latencies, 99)), 3),
        "pass_nfr": np.percentile(latencies, 95) <= NFR_LIMIT_MS,
    }


def main():
    ONNX_BASE.mkdir(parents=True, exist_ok=True)
    results = []

    for model_key, hf_path in MODELS.items():
        for max_len in TOKEN_LENGTHS:
            print(f"\n{'='*50}")
            print(f"Model: {model_key} | Token: {max_len}")

            use_quantization = model_key not in {"xlmr-fp32", "t5-small"}
            quant_type = "INT8" if use_quantization else "FP32"

            try:
                export_dir = export_onnx(model_key, hf_path, max_len, quantize=use_quantization)
                metrics    = benchmark(export_dir, max_len)

                row = {
                    "model":    model_key,
                    "max_len":  max_len,
                    "quant":    quant_type,
                    "avg_ms":   metrics["avg_ms"],
                    "p95_ms":   metrics["p95_ms"],
                    "p99_ms":   metrics["p99_ms"],
                    "pass_nfr": "✓" if metrics["pass_nfr"] else "✗ NFR FAIL",
                }
                results.append(row)
                print(f"  avg={metrics['avg_ms']}ms | p95={metrics['p95_ms']}ms | NFR={'PASS' if metrics['pass_nfr'] else 'FAIL'}")

            except Exception as e:
                print(f"  [ERROR] {model_key} tok={max_len}: {e}")
                results.append({
                    "model": model_key, "max_len": max_len,
                    "quant": quant_type, "avg_ms": -1,
                    "p95_ms": -1, "p99_ms": -1, "pass_nfr": "ERROR"
                })

    # CSV'ye yaz
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model","max_len","quant","avg_ms","p95_ms","p99_ms","pass_nfr"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Ablasyon tamamlandı → {OUTPUT_CSV}")
    _print_table(results)


def _print_table(results):
    print(f"\n{'Model':<18} {'Tok':>4} {'Quant':>5} {'avg_ms':>8} {'p95_ms':>8} {'p99_ms':>8}  NFR")
    print("-" * 65)
    for r in results:
        nfr = r['pass_nfr']
        print(f"{r['model']:<18} {r['max_len']:>4} {r['quant']:>5} "
              f"{r['avg_ms']:>8} {r['p95_ms']:>8} {r['p99_ms']:>8}  {nfr}")


if __name__ == "__main__":
    main()
