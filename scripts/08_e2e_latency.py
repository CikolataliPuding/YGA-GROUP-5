# scripts/08_e2e_latency.py
"""
E2E Latency Ablasyonu: Farklı model konfigürasyonlarında
uçtan uca gecikme ölçümü.
Akış: Fast Masker → ONNX Inference → Rule Engine → Karar
"""

import sys
import time
import json
import random
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RANDOM_SEED = 42
SAMPLE_SIZE = 200
WARMUP = 20
DATASET_PATH = Path("data/processed/test.jsonl")
API_URL = "http://127.0.0.1:8000/classify"

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
                text = obj.get("text") or obj.get("prompt") or obj.get("input", "")
                if text:
                    samples.append(text)
    random.seed(seed)
    return random.sample(samples, min(n, len(samples)))

# ============================================================
# 2. Bileşen bazlı ölçüm
# ============================================================
def benchmark_components(samples: list[str], warmup: int) -> dict:
    from masking.fast_masker import mask_text
    from app.classifier import ErlikClassifier

    classifier = ErlikClassifier()

    # Warmup
    for s in samples[:warmup]:
        mask_text(s)
        classifier.classify(s)

    masking_times = []
    inference_times = []
    e2e_times = []

    for s in samples:
        # E2E başlangıç
        t_e2e = time.perf_counter()

        # Masking
        t0 = time.perf_counter()
        masked, _ = mask_text(s)
        masking_times.append((time.perf_counter() - t0) * 1000)

        # Inference
        t0 = time.perf_counter()
        result = classifier.classify(masked)
        inference_times.append((time.perf_counter() - t0) * 1000)

        # E2E bitiş
        e2e_times.append((time.perf_counter() - t_e2e) * 1000)

    def stats(times):
        s = sorted(times)
        return {
            "avg_ms": round(sum(s) / len(s), 4),
            "p50_ms": round(s[len(s) // 2], 4),
            "p95_ms": round(s[int(len(s) * 0.95)], 4),
            "max_ms": round(s[-1], 4),
        }

    return {
        "masking":   stats(masking_times),
        "inference": stats(inference_times),
        "e2e":       stats(e2e_times),
    }

# ============================================================
# 3. API üzerinden ölçüm (HTTP overhead dahil)
# ============================================================
def benchmark_api(samples: list[str], warmup: int) -> dict:
    # Warmup
    for s in samples[:warmup]:
        try:
            requests.post(API_URL, json={"text": s}, timeout=5)
        except:
            pass

    times = []
    e2e_values = []

    for s in samples:
        t0 = time.perf_counter()
        try:
            r = requests.post(API_URL, json={"text": s}, timeout=5)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            data = r.json()
            if "e2e_ms" in data:
                e2e_values.append(data["e2e_ms"])
        except Exception as e:
            print(f"API hatasi: {e}")

    s = sorted(times)
    e = sorted(e2e_values) if e2e_values else []

    return {
        "http_total": {
            "avg_ms": round(sum(s) / len(s), 4),
            "p50_ms": round(s[len(s) // 2], 4),
            "p95_ms": round(s[int(len(s) * 0.95)], 4),
            "max_ms": round(s[-1], 4),
        },
        "e2e_internal": {
            "avg_ms": round(sum(e) / len(e), 4) if e else None,
            "p50_ms": round(e[len(e) // 2], 4) if e else None,
            "p95_ms": round(e[int(len(e) * 0.95)], 4) if e else None,
            "max_ms": round(e[-1], 4) if e else None,
        } if e else None
    }

# ============================================================
# 4. Ana akış
# ============================================================
if __name__ == "__main__":
    print(f"Dataset yukleniyor: {DATASET_PATH}")
    samples = load_samples(DATASET_PATH, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Ornekler: {len(samples)} | Warmup: {WARMUP} | Seed: {RANDOM_SEED}")
    print("=" * 70)

    # Bileşen bazlı ölçüm
    print("\n[1] BILESEN BAZLI OLCUM (dogrudan Python)...")
    comp = benchmark_components(samples, WARMUP)

    print(f"\n{'Bileşen':<15} {'avg':>10} {'p50':>10} {'p95':>10} {'max':>10}")
    print("-" * 55)
    for name, stats in comp.items():
        print(f"{name:<15} {stats['avg_ms']:>10.4f} {stats['p50_ms']:>10.4f} "
              f"{stats['p95_ms']:>10.4f} {stats['max_ms']:>10.4f}")

    # API ölçümü
    print(f"\n[2] API OLCUMU (HTTP overhead dahil)...")
    print("FastAPI'nin calisiyor olmasi gerekiyor: uvicorn app.main:app --reload")
    
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=2)
        api = benchmark_api(samples, WARMUP)
        
        print(f"\n{'Ölçüm':<20} {'avg':>10} {'p50':>10} {'p95':>10} {'max':>10}")
        print("-" * 60)
        
        h = api["http_total"]
        print(f"{'HTTP Total':<20} {h['avg_ms']:>10.4f} {h['p50_ms']:>10.4f} "
              f"{h['p95_ms']:>10.4f} {h['max_ms']:>10.4f}")
        
        if api.get("e2e_internal"):
            e = api["e2e_internal"]
            print(f"{'E2E Internal':<20} {e['avg_ms']:>10.4f} {e['p50_ms']:>10.4f} "
                  f"{e['p95_ms']:>10.4f} {e['max_ms']:>10.4f}")
    except:
        print("FastAPI cevap vermiyor, API olcumu atlaniyor.")
        api = None

    # NFR-01 özeti
    print("\n" + "=" * 70)
    print("NFR-01 OZETI (p95 <= 10ms model cikarimi, <= 50ms E2E)")
    print("=" * 70)
    
    inf_p95 = comp["inference"]["p95_ms"]
    e2e_p95 = comp["e2e"]["p95_ms"]
    
    print(f"Model Cikarim p95 : {inf_p95:.4f}ms → "
          f"{'PASS' if inf_p95 <= 10 else 'FAIL'}")
    print(f"E2E p95           : {e2e_p95:.4f}ms → "
          f"{'PASS' if e2e_p95 <= 50 else 'FAIL'}")

    # Kaydet
    output = {
        "seed": RANDOM_SEED,
        "sample_size": len(samples),
        "warmup": WARMUP,
        "components": comp,
        "api": api,
    }
    
    out_path = Path("docs/e2e_latency.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSonuclar kaydedildi: {out_path}")