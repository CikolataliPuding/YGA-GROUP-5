# scripts/inspect_dataset.py

import json

guvenli, kaytarma, tehdit = [], [], []

for f in ["data/processed/train.jsonl", "data/processed/val.jsonl", "data/processed/test.jsonl"]:
    with open(f, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["label"] == 0:
                guvenli.append(obj["text"])
            elif obj["label"] == 1:
                kaytarma.append(obj["text"])
            else:
                tehdit.append(obj["text"])

print(f"Toplam — GUVENLI: {len(guvenli)} | KAYTARMA: {len(kaytarma)} | TEHDIT: {len(tehdit)}\n")

print("GUVENLI (ilk 10):")
for t in guvenli[:10]:
    print(f"  {t}")

print("\nKAYTARMA (ilk 10):")
for t in kaytarma[:10]:
    print(f"  {t}")

print("\nTEHDIT (ilk 10):")
for t in tehdit[:10]:
    print(f"  {t}")