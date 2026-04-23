# scripts/04_train.py

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, classification_report

# ── CONFIG ──────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH   = "data/processed/val.jsonl"
TEST_PATH  = "data/processed/test.jsonl"
MAX_LENGTH = 64
BATCH_SIZE = 16
LR         = 2e-5
SEED       = 42

ID2LABEL = {0: "GUVENLI", 1: "KAYTARMA", 2: "TEHDIT"}
LABEL2ID = {"GUVENLI": 0, "KAYTARMA": 1, "TEHDIT": 2}
# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="xlm-roberta-base")
    parser.add_argument("--output", type=str, default="checkpoints/xlmr-erlikgate")
    parser.add_argument("--epochs", type=int, default=6)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({
                "text":  obj["text"],
                "label": int(obj["label"]),
            })
    return records


def tokenize_fn(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro")}


def main():
    args       = parse_args()
    MODEL_NAME = args.model
    OUTPUT_DIR = args.output
    EPOCHS     = args.epochs

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Output  : {OUTPUT_DIR}\n")

    from collections import Counter
    train_records = load_jsonl(TRAIN_PATH)
    val_records   = load_jsonl(VAL_PATH)
    test_records  = load_jsonl(TEST_PATH)

    all_records = train_records + val_records + test_records
    print(f"Toplam ornek: {len(all_records)}")
    dist = Counter(r["label"] for r in all_records)
    for lbl, cnt in sorted(dist.items()):
        print(f"  {ID2LABEL[lbl]:<10} : {cnt} ornek")

    train_ds = Dataset.from_dict({
        "text":  [r["text"]  for r in train_records],
        "label": [r["label"] for r in train_records],
    })
    val_ds = Dataset.from_dict({
        "text":  [r["text"]  for r in val_records],
        "label": [r["label"] for r in val_records],
    })
    test_ds = Dataset.from_dict({
        "text":  [r["text"]  for r in test_records],
        "label": [r["label"] for r in test_records],
    })
    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    fn = lambda batch: tokenize_fn(batch, tokenizer)
    train_ds = train_ds.map(fn, batched=True)
    val_ds   = val_ds.map(fn, batched=True)
    test_ds  = test_ds.map(fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n── Test seti değerlendirmesi ──")
    preds_out = trainer.predict(test_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    print(classification_report(
        preds_out.label_ids, preds,
        target_names=[ID2LABEL[i] for i in range(3)],
    ))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel kaydedildi → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()