from datasets import load_dataset
import json
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        "name": "deepset/prompt-injections",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "label_map": {0: 0, 1: 2}
    },
    {
        "name": "JasperLS/prompt-injections",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "label_map": {0: 0, 1: 2}
    },
    {
        "name": "rubend18/ChatGPT-Jailbreak-Prompts",
        "split": "train",
        "text_col": "Prompt",
        "label_col": None,
        "label_map": None
    },
]

total = {0: 0, 2: 0}

for src in SOURCES:
    print(f"\nIndiriliyor: {src['name']}")
    try:
        ds = load_dataset(src["name"], split=src["split"])
    except Exception as e:
        print(f"  HATA: {e}")
        continue

    safe_name = src["name"].replace("/", "_")
    out_path  = Path("data/raw") / f"{safe_name}.jsonl"
    count     = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = str(row.get(src["text_col"], "")).strip()
            if len(text) < 5:
                continue

            if src["label_col"] is None:
                erg_label = 2
            else:
                hf_label  = row[src["label_col"]]
                erg_label = src["label_map"].get(hf_label, -1)
                if erg_label == -1:
                    continue

            record = {
                "text": text,
                "label": erg_label,
                "source": src["name"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if erg_label in total:
                total[erg_label] += 1

    print(f"  → {count} ornek kaydedildi")

print(f"\n=== SONUC ===")
print(f"GUVENLI(0):  {total[0]}")
print(f"TEHDIT(2):   {total[2]}")