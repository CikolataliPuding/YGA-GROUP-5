# app/classifier.py

import time
import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from gateway.rule_engine import kural_motoru
# ── CONFIG ───────────────────────────────────────────────────────────────────
ONNX_DIR   = "onnx/minilm-int8-tok32"
FILE_NAME  = "model_quantized.onnx"
MAX_LENGTH = 32
CONFIDENCE_THRESHOLD = 0.75

ID2LABEL = {0: "GUVENLI", 1: "KAYTARMA", 2: "TEHDIT"}
# ─────────────────────────────────────────────────────────────────────────────


class ErlikClassifier:
    def __init__(self):
        self.model = ORTModelForSequenceClassification.from_pretrained(
            ONNX_DIR,
            file_name=FILE_NAME,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            ONNX_DIR,
            fix_mistral_regex=True,
        )
        self.input_names = {
            inp.name for inp in self.model.model.get_inputs()
        }

    # ── Katman 1: Model ──────────────────────────────────────────────────────
    def _model_predict(self, text: str) -> tuple[str, float, float]:
        inp = self.tokenizer(
            text,
            return_tensors="np",
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
        )
        # Sadece modelin beklediği girdileri geçir; fazladan anahtarlar runtime hatası verebilir
        inp = {k: v for k, v in inp.items() if k in self.input_names}

        t0  = time.perf_counter()
        out = self.model(**inp)
        ms  = (time.perf_counter() - t0) * 1000

        probs      = self._softmax(out.logits[0])
        label_id   = int(np.argmax(probs))
        confidence = float(probs[label_id])
        return ID2LABEL[label_id], confidence, ms

    # ── Katman 3: Karar Matrisi ───────────────────────────────────────────────
    def _karar_matrisi(
        self,
        model_label: str,
        confidence: float,
        rule_label: str | None,
        matched_rule: str | None,
    ) -> tuple[str, str]:
        """(decision, source) döndürür."""

        if rule_label is None:
            return model_label, "model"

        if rule_label == "TEHDIT":
            return "TEHDIT", "rule"

        if rule_label == "KAYTARMA":
            return "KAYTARMA", "rule"

        # rule_label == "GUVENLI"
        if model_label == "GUVENLI":
            return "GUVENLI", "rule+model"

        # Çelişki: kural GUVENLI, model farklı
        if confidence >= CONFIDENCE_THRESHOLD:
            return model_label, "model"
        return "GUVENLI", "rule"

    def classify(self, text: str) -> dict:
        text = text.strip()

        # Model için: TR-özel normalizasyon (I→ı dönüşümü sadece model girdisine uygulanır)
        text_normalized = (text.replace("I", "ı")
                               .replace("İ", "i")
                               .replace("Ş", "ş")
                               .replace("Ğ", "ğ")
                               .replace("Ü", "ü")
                               .replace("Ö", "ö")
                               .replace("Ç", "ç"))
        text_normalized = text_normalized.lower()

        # Katman 1 — Model (TR-normalize edilmiş metin)
        model_label, confidence, inference_ms = self._model_predict(text_normalized)

        # Katman 2 — Kural Motoru (orijinal metin; içeride iki normalizasyon uygulanır)
        rule_label, _, matched_rule = kural_motoru(text)

        # Katman 3 — Karar Matrisi
        decision, source = self._karar_matrisi(
            model_label, confidence, rule_label, matched_rule
        )

        # `confidence` alanı nihai karara ait olmalı; model güveni ayrı döndürülür.
        decision_confidence = confidence if decision == model_label else 1.0

        return {
            "decision":         decision,
            "label":            model_label,
            "confidence":       round(decision_confidence, 4),
            "model_confidence": round(confidence, 4),
            "inference_ms":     round(inference_ms, 3),
            "source":           source,
            "rule_match":       matched_rule,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()