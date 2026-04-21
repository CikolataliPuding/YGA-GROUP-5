import json
from pathlib import Path

Path("data/manual").mkdir(parents=True, exist_ok=True)

def txt_to_jsonl(txt_path, label, out_path):
    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        text = line.strip()
        # Boş satır, başlık veya kategori satırlarını atla
        if not text:
            continue
        if text.startswith("Niyet:"):
            continue
        if text[0].isdigit() and "." in text[:3]:
            continue

        records.append({
            "text": text,
            "label": label,
            "source": "manual"
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"→ {out_path}: {len(records)} ornek")

txt_to_jsonl("data/manual/IS.txt",       0, "data/manual/guvenli.jsonl")
txt_to_jsonl("data/manual/KAYTARMA.txt", 1, "data/manual/kaytarma.jsonl")