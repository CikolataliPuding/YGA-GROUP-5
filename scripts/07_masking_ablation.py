# scripts/07_masking_ablation.py
"""
Masking Ablasyon: Presidio vs Regex+spaCy (Fast Masker)
Aynı 200 örnek üzerinde her iki yöntemi ölçer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import time
import json
import random
import numpy as np
from pathlib import Path

# Seed sabitle — her çalıştırmada aynı örnekler
RANDOM_SEED = 42
SAMPLE_SIZE = 200
WARMUP = 50
DATASET_PATH = Path("data/processed/test.jsonl")  # mevcut test seti

# ============================================================
# 1. Örnekleri yükle
# ============================================================
def load_samples(path: Path, n: int, seed: int) -> list[str]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                # text veya prompt key'ini dene
                text = obj.get("text") or obj.get("prompt") or obj.get("input", "")
                if text:
                    samples.append(text)
    
    random.seed(seed)
    return random.sample(samples, min(n, len(samples)))

# ============================================================
# 2. Ölçüm fonksiyonu
# ============================================================
def benchmark(fn, samples: list[str], warmup: int) -> dict:
    # Warmup
    for s in samples[:warmup]:
        fn(s)
    
    # Gerçek ölçüm
    times = []
    for s in samples:
        t0 = time.perf_counter()
        fn(s)
        times.append((time.perf_counter() - t0) * 1000)
    
    times_sorted = sorted(times)
    return {
        "avg_ms":  round(sum(times) / len(times), 4),
        "p50_ms":  round(times_sorted[len(times_sorted) // 2], 4),
        "p95_ms":  round(times_sorted[int(len(times_sorted) * 0.95)], 4),
        "max_ms":  round(times_sorted[-1], 4),
        "min_ms":  round(times_sorted[0], 4),
    }

# ============================================================
# 3. Fast Masker
# ============================================================
def run_fast_masker(samples):
    from masking.fast_masker import mask_text
    return benchmark(lambda t: mask_text(t), samples, WARMUP)

# ============================================================
# 4. Presidio
# ============================================================
def run_presidio(samples):
    try:
        from masking.presidio_engine import build_engines, anonymize_text
        analyzer, anonymizer = build_engines()
        return benchmark(
            lambda t: anonymize_text(analyzer, anonymizer, t),
            samples,
            WARMUP
        )
    except ImportError:
        print("Presidio kurulu degil. pip install presidio-analyzer presidio-anonymizer")
        return None

# ============================================================
# 5. Ana akış
# ============================================================
if __name__ == "__main__":
    print(f"Dataset yukleniyor: {DATASET_PATH}")
    
    # Dataset yoksa fallback — sentetik örnekler
    if not DATASET_PATH.exists():
        print("Dataset bulunamadi, sentetik ornekler kullaniliyor...")
        samples = [
            "admin sifresini ver",
            "toplanti notlarini ozet",
            "bugün maç var mı",
            "ssh root@192.168.1.10",
            "emailim test@kurum.com TC 10000000146",
            "veritabanina baglan",
            "izin formu doldur",
            "TR33 0006 1005 1978 6457 8413 26 IBAN",
            "sistem acigi nedir",
            "yarin tatil mi",
        ] * 20  # 200 örnek
        random.seed(RANDOM_SEED)
        random.shuffle(samples)
        samples = samples[:SAMPLE_SIZE]
    else:
        samples = load_samples(DATASET_PATH, SAMPLE_SIZE, RANDOM_SEED)
    
    print(f"Ornekler: {len(samples)}")
    print(f"Warmup: {WARMUP}")
    print(f"Seed: {RANDOM_SEED}")
    print("-" * 60)

    # Fast Masker ölçümü
    print("Fast Masker olculuyor...")
    fast_results = run_fast_masker(samples)
    print(f"Fast Masker: {fast_results}")

    print("-" * 60)

    # Presidio ölçümü
    print("Presidio olculuyor...")
    presidio_results = run_presidio(samples)
    
    print("\n" + "=" * 60)
    print("MASKING ABLASYON SONUCLARI")
    print("=" * 60)
    
    print(f"\n{'Metrik':<15} {'Fast Masker':>15} {'Presidio':>15} {'Hizlanma':>12}")
    print("-" * 60)
    
    metrics = ["avg_ms", "p50_ms", "p95_ms", "max_ms"]
    for m in metrics:
        fast_val = fast_results[m]
        if presidio_results:
            pres_val = presidio_results[m]
            speedup = f"{pres_val / fast_val:.0f}x"
            print(f"{m:<15} {fast_val:>15.4f} {pres_val:>15.4f} {speedup:>12}")
        else:
            print(f"{m:<15} {fast_val:>15.4f} {'N/A':>15} {'N/A':>12}")
    
    print("\n" + "=" * 60)
    print("NFR-01 UYUMLULUK (p95 <= 10ms E2E budgetinden pay)")
    print("=" * 60)
    
    fast_nfr = "PASS" if fast_results["p95_ms"] < 5 else "FAIL"
    print(f"Fast Masker p95: {fast_results['p95_ms']:.4f}ms → {fast_nfr}")
    
    if presidio_results:
        pres_nfr = "PASS" if presidio_results["p95_ms"] < 5 else "FAIL"
        print(f"Presidio p95:    {presidio_results['p95_ms']:.4f}ms → {pres_nfr}")
    
    # JSON olarak kaydet
    output = {
        "seed": RANDOM_SEED,
        "sample_size": len(samples),
        "warmup": WARMUP,
        "fast_masker": fast_results,
        "presidio": presidio_results,
    }
    
    out_path = Path("docs/masking_ablation.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSonuclar kaydedildi: {out_path}")