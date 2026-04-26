# scripts/check_lengths.py

import json

records = []
for f in ["data/processed/train.jsonl", "data/processed/val.jsonl", "data/processed/test.jsonl"]:
    with open(f, encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                records.append(json.loads(line))

lengths = [len(r["text"].split()) for r in records]
print(f"Toplam örnek  : {len(records)}")
print(f"Ortalama kelime: {sum(lengths)//len(lengths)}")
print(f"Max kelime     : {max(lengths)}")
print(f"Min kelime     : {min(lengths)}")
print(f"64 tokeni aşan : {sum(1 for l in lengths if l > 45)} ({sum(1 for l in lengths if l > 45)*100//len(lengths)}%)")