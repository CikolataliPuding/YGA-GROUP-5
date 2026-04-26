# app/classifier.py

import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from gateway import kural_motoru
# ── CONFIG ───────────────────────────────────────────────────────────────────
ONNX_PATH  = "onnx/minilm-int8-tok32/model_quantized.onnx"
MODEL_DIR  = "onnx/minilm-int8-tok32"
MAX_LENGTH = 32
CONFIDENCE_THRESHOLD = 0.75

ID2LABEL = {0: "GUVENLI", 1: "KAYTARMA", 2: "TEHDIT"}
# ─────────────────────────────────────────────────────────────────────────────


class ErlikClassifier:
    def __init__(self):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            ONNX_PATH,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, fix_mistral_regex=True
        )

        # ONNX modelinin beklediği input isimlerini al
        self.input_names = {
            inp.name for inp in self.session.get_inputs()
        }

    # ── Katman 1: Model ──────────────────────────────────────────────────────
    def _model_predict(self, text: str) -> tuple[str, float, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
        )

        # Sadece modelin beklediği input'ları gönder
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = inputs["input_ids"].astype(np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = inputs["attention_mask"].astype(np.int64)
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        t0     = time.perf_counter()
        logits = self.session.run(None, feed)[0][0]
        ms     = (time.perf_counter() - t0) * 1000

        probs      = self._softmax(logits)
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
        # Katman 1 — Model
        model_label, confidence, inference_ms = self._model_predict(text)

        # Katman 2 — Kural Motoru
        rule_label, _, matched_rule = kural_motoru(text)

        # Katman 3 — Karar Matrisi
        decision, source = self._karar_matrisi(
            model_label, confidence, rule_label, matched_rule
        )

        return {
            "decision":     decision,
            "label":        model_label,
            "confidence":   round(confidence, 4),
            "inference_ms": round(inference_ms, 3),
            "source":       source,
            "rule_match":   matched_rule,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()