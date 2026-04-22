# ErlikGate — Model & Sistem Dokümantasyonu

## 1. Geliştirme Ortamı

| Parametre | Değer |
|-----------|-------|
| İşletim Sistemi | Windows 11 |
| İşlemci | Intel Core i5-12500H (12. Nesil) |
| SIMD Desteği | AVX2 (AVX512 yok) |
| GPU | CUDA 12.4 (eğitim için) |
| Python | 3.13 |
| PyTorch | 2.6.0+cu124 |
| ONNX Runtime | optimum[onnxruntime] |

---

## 2. Temel Model

| Parametre | Değer |
|-----------|-------|
| Model | `xlm-roberta-base` |
| Geliştirici | Facebook AI (Meta) |
| Parametre Sayısı | 125M |
| Mimari | Encoder-only Transformer (BERT ailesi) |
| Dil Desteği | 100 dil (Türkçe dahil) |
| Pre-train Verisi | 2.5TB Common Crawl |
| Görev | Sequence Classification (3 sınıf) |

**Seçilme Gerekçesi:**
- Encoder-only mimari → NFR-01 uyumlu (üretken model değil)
- Türkçe destek → dataset'in bir kısmı Türkçe
- `base` boyutu → `large`'a göre 2x hızlı, yeterli doğruluk
- ONNX ekosistemi → production-ready export

---

## 3. Fine-Tune Detayları

| Parametre | Değer |
|-----------|-------|
| Base Model | `xlm-roberta-base` |
| Dataset | 300 örnek, dengeli 3 sınıf |
| Sınıflar | GUVENLI (0), KAYTARMA (1), TEHDIT (2) |
| Split | Train: 210 / Val: 45 / Test: 45 |
| Epochs | 6 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Max Length | 64 token |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Early Stopping | Patience: 3 |
| Mixed Precision | fp16 (CUDA aktifken) |

**Dataset Kaynakları:**
- `deepset/prompt-injections` (HuggingFace)
- Manuel örnekler (KAYTARMA + GUVENLI + TEHDIT)

---

## 4. ONNX Export & Quantization

| Adım | Detay |
|------|-------|
| Export | PyTorch → ONNX FP32 (`optimum`) |
| Quantization | Dynamic INT8 |
| Quant Config | `AutoQuantizationConfig.avx2` |
| SIMD Hedef | AVX2 (i5-12500H uyumlu) |
| Thread Ayarı | intra=4, inter=1, ORT_SEQUENTIAL |
| Graph Opt. | ORT_ENABLE_ALL |

**Not:** İlk export'ta `avx512_vnni` kullanıldı, i5-12500H'da AVX512 olmadığı için `avx2`'ye geçildi. AVX2 ile hem doğruluk hem hız iyileşti.

---

## 5. Test Seti Sonuçları

### INT8 Modeli (Production)

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|----|
| GUVENLI | 0.9412 | 0.9412 | 0.9412 |
| KAYTARMA | 0.9167 | 1.0000 | 0.9565 |
| TEHDIT | 1.0000 | 0.9412 | 0.9697 |
| **macro** | **0.9526** | **0.9608** | **0.9558** |
| **accuracy** | | | **0.9556** |

### FP32 Modeli (Referans)

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|----|
| GUVENLI | 1.0000 | 0.9412 | 0.9697 |
| KAYTARMA | 0.9167 | 1.0000 | 0.9565 |
| TEHDIT | 1.0000 | 1.0000 | 1.0000 |
| **macro** | **0.9722** | **0.9804** | **0.9754** |
| **accuracy** | | | **0.9778** |

**Quantization Etkisi:**
- Hız kazancı: ~2.5x (p95: 25ms → 10ms)
- Doğruluk kaybı: %2.2 (kabul edilebilir)

---

## 6. Latency Ölçümleri

### INT8 vs FP32 (max_length=32, Windows CPU)

| Model | avg_ms | p50_ms | p95_ms | NFR-01 |
|-------|--------|--------|--------|--------|
| INT8 | 8.916 | 8.750 | 10.146 | ~ |
| FP32 | 22.443 | 22.022 | 25.152 | FAIL |

### Token Ablasyonu (INT8, Windows CPU)

| max_length | accuracy | f1_macro | p50_ms | p95_ms | NFR-01 |
|------------|----------|----------|--------|--------|--------|
| 32 | 0.9556 | 0.9558 | 9.353 | 11.164 | ~ |
| 64 | 0.9556 | 0.9558 | 15.338 | 17.939 | FAIL |
| 128 | 0.9556 | 0.9558 | 26.283 | 30.598 | FAIL |
| 256 | 0.9556 | 0.9558 | 50.852 | 61.396 | FAIL |

**NFR-01 Notu:**
> Ölçümler Windows 11 / i5-12500H ortamında yapılmıştır. Linux/AVX2 üretim ortamında p95 değerlerinin 2-3x düşmesi beklenmektedir (tok=64 için ~6ms hedefi).

---

## 7. Mimari Karar Gerekçesi

### Neden max_length=64?

Token ablasyonu sonuçlarına göre doğruluk tüm uzunluklarda sabit (0.9556). Bu, prompt'larımızın 32 token ile yeterince temsil edilebildiğini göstermektedir. Ancak:

- `max_length=32` → bazı uzun prompt'lar kesilebilir
- `max_length=64` → güvenli kapsam, makul gecikme
- `max_length=128+` → gereksiz yavaş, doğruluk kazancı yok

**Sonuç:** max_length=64, doğruluk ve gecikme arasındaki optimal denge noktasıdır.



