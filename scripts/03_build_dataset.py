import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

# --- Tum dosyalari yukle ---
sources = [
    # HF verileri
    "data/raw/deepset_prompt-injections.jsonl",
    "data/raw/JasperLS_prompt-injections.jsonl",
    "data/raw/rubend18_ChatGPT-Jailbreak-Prompts.jsonl",
    # Manuel veriler
    "data/manual/guvenli.jsonl",
    "data/manual/kaytarma.jsonl",
]

records = []
for path in sources:
    p = Path(path)
    if not p.exists():
        print(f"UYARI: {path} bulunamadi, atlaniyor.")
        continue
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

print(f"Ham toplam: {len(records)}")
print(f"Dagilim: {Counter(r['label'] for r in records)}")

# --- Sinifa gore ayir ---
by_label = {0: [], 1: [], 2: []}
for r in records:
    if r["label"] in by_label:
        by_label[r["label"]].append(r)

# --- Her siniftan 100 ornek al ---
TARGET = 100
balanced = []
for label, samples in by_label.items():
    random.shuffle(samples)
    selected = samples[:TARGET]
    balanced.extend(selected)
    print(f"  Label {label}: {len(selected)} ornek secildi")

random.shuffle(balanced)

# --- Split: %70 train / %15 val / %15 test ---
n      = len(balanced)
n_test = int(n * 0.15)
n_val  = int(n * 0.15)

test  = balanced[:n_test]
val   = balanced[n_test:n_test + n_val]
train = balanced[n_test + n_val:]

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
