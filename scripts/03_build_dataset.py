import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

# --- Tum dosyalari yukle ---
sources = [
    #     # Manuel veriler
    "data/raw/erlikgate_v3.jsonl",
    "data/raw/erlikgate_v2.jsonl",
    "data/raw/erlikgate_v4.jsonl",
    "data/manual/guvenli.jsonl",
    "data/raw/erlikgate_synthetic_v1.jsonl",
    "data/raw/erlikgate_synthetic_v2.jsonl",
    "data/manual/kaytarma.jsonl",
    "data/raw/akademik_ek.jsonl",
]

records = []
for path in sources:
    p = Path(path)
    if not p.exists():
        print(f"UYARI: {path} bulunamadi, atlaniyor.")
        continue
    source_count = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                source_count += 1
    if source_count == 0:
        raise ValueError(
            f"HATA: {path} mevcut ama 0 kayit uretti. "
            f"Kaynak dosya bos veya gecersiz olabilir."
        )
    print(f"Kaynak yuklendi: {path} ({source_count} kayit)")

print(f"Ham toplam: {len(records)}")
print(f"Dagilim: {Counter(r['label'] for r in records)}")

# --- Deduplikasyon ---
seen = set()
deduped = []
for r in records:
    normalized = r["text"].strip().lower()
    if normalized not in seen:
        seen.add(normalized)
        deduped.append(r)

print(f"Deduplikasyon sonrasi: {len(deduped)} ornek "
      f"({len(records) - len(deduped)} tekrar silindi)")

# --- Sinifa gore ayir ---
by_label = {0: [], 1: [], 2: []}
for r in deduped:
    if r["label"] in by_label:
        by_label[r["label"]].append(r)

# --- Her siniftan min(label_count) kadar ornek al ---
TARGET = min(len(samples) for samples in by_label.values())
print(f"Her siniftan {TARGET} ornek alinacak")
balanced = []
for label, samples in by_label.items():
    random.shuffle(samples)
    selected = samples[:TARGET]
    balanced.extend(selected)
    print(f"  Label {label}: {len(selected)} ornek secildi")

def stratified_split(data, test_ratio=0.15, val_ratio=0.15):
    grouped = {}
    for record in data:
        label = record["label"]
        grouped.setdefault(label, []).append(record)

    train, val, test = [], [], []

    for samples in grouped.values():
        random.shuffle(samples)
        n = len(samples)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        test.extend(samples[:n_test])
        val.extend(samples[n_test:n_test + n_val])
        train.extend(samples[n_test + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test

# --- Split: %70 train / %15 val / %15 test (stratified) ---
train, val, test = stratified_split(balanced, test_ratio=0.15, val_ratio=0.15)
# --- Kaydet ---
Path("data/processed").mkdir(parents=True, exist_ok=True)

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Kaydedildi: {path} ({len(data)} ornek)")

save_jsonl(train, "data/processed/train.jsonl")
save_jsonl(val,   "data/processed/val.jsonl")
save_jsonl(test,  "data/processed/test.jsonl")
