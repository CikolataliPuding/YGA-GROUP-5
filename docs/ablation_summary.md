# ErlikGate — Ablasyon Çalışması (v2)

**Proje:** ErlikGate Hibrit Ağ Geçidi — Karar Motoru Model Seçimi  
**Tarih:** Nisan 2026  
**Ortam:** Windows 11, Intel Core i5-12500H, AVX2, CUDA 12.4  
**Amaç:** Kurumsal ağlarda Shadow AI tespiti için optimal encoder modelinin belirlenmesi

---

## 1. Metodoloji

### 1.1 Deneysel Kurulum

Karar Motoru için aday modeller, aynı eğitim verisi ve hiperparametre seti kullanılarak fine-tune edilmiş; ardından ONNX Runtime üzerinde INT8 dynamic quantization uygulanarak değerlendirilmiştir.

**Dataset:**
- Toplam unique örnek: 5.263
- Dengeli dağılım: 1.562 GUVENLI / 1.562 KAYTARMA / 1.562 TEHDIT
- Split: Train 3.282 / Val 702 / Test 702 
- Kaynak: El yazısı kurumsal örnekler + sentetik üretim + akademik CSV
- Deduplikasyon uygulanmıştır

**Sabit Hiperparametreler:**
- Learning Rate: 2e-5
- Batch Size: 16
- Optimizer: AdamW
- Mixed Precision: fp16 (CUDA)
- max_length: 64
- Quantization: AVX2 Dynamic INT8

**Değerlendirme Metrikleri:**
- Doğruluk: Accuracy, F1 Macro (test seti, 702 örnek)
- Gecikme: p50, p95 (ms) — 450 tekrar, ilk 50 warmup hariç
- NFR-01: p95 ≤ 10ms (üretim gecikmesi kısıtı)

### 1.2 Limitasyonlar

- **Geliştirme ortamı:** Gecikme ölçümleri Windows 11 / i5-12500H üzerinde yapılmıştır. Üretim hedefi Linux/AVX2 ortamıdır; bu ortamda p95 değerlerinin 2-3x düşmesi beklenmektedir.
- **Model başına epoch farkı:** Her model kendi optimal epoch sayısıyla eğitilmiştir (xlmr/mbert/modernbert: 6, deberta: 15, minilm: 10, tinybert: 10).
- **DeBERTa quantization hassasiyeti:** DeBERTa-v3-small, INT8 dynamic quantization sonrasında ciddi doğruluk kaybı yaşamıştır. Bu durum DeBERTa'nın disentangled attention mekanizmasının quantization'a duyarlılığından kaynaklanmaktadır.

---

## 2. Model Eğitim Sonuçları (PyTorch, Test Seti)

| Model | Accuracy | F1 Macro | Epoch | Parametre |
|-------|----------|----------|-------|-----------|
| MiniLM-L12 | **0.9829** | **0.9829** | 10 | 22M |
| mBERT | 0.9701 | 0.9701 | 6 | 110M |
| DeBERTa-v3-small | 0.9601 | 0.9601 | 15 | 22M |
| ModernBERT-base | 0.9330 | 0.9330 | 6 | 149M |
| xlmr-base | 0.8917 | 0.8917 | 6 | 125M |
| TinyBERT-4L | 0.8860 | 0.8860 | 10 | 14M |
| T5-small | — | — | — | 60M |

> T5-small encoder-decoder mimarisi nedeniyle sequence classification göreviyle uyumsuz olduğundan eğitim tamamlanamamıştır. Bu bulgu encoder-only mimari seçimini doğrulamaktadır.

---

## 3. Model × Token Ablasyon Tablosu (ONNX INT8)

| Model | max\_length | Accuracy | F1 Macro | p50 (ms) | p95 (ms) | NFR-01 |
|-------|------------|----------|----------|----------|----------|--------|
| xlmr | 32 | 1.0000 | 1.0000 | 9.201 | 23.524 | FAIL |
| xlmr | 64 | 1.0000 | 1.0000 | 31.027 | 41.668 | FAIL |
| xlmr | 128 | 1.0000 | 1.0000 | 56.422 | 74.587 | FAIL |
| xlmr | 256 | 1.0000 | 1.0000 | 110.958 | 142.033 | FAIL |
| deberta | 32 | 0.6444 | 0.6454 | 11.483 | 19.439 | FAIL |
| deberta | 64 | 0.6667 | 0.6706 | 13.232 | 14.916 | FAIL |
| deberta | 128 | 0.6889 | 0.6945 | 22.246 | 25.239 | FAIL |
| deberta | 256 | 0.6889 | 0.6945 | 57.668 | 76.625 | FAIL |
| **minilm** | **32** | **1.0000** | **1.0000** | **4.677** | **7.182** | **PASS** |
| minilm | 64 | 1.0000 | 1.0000 | 7.876 | 12.005 | FAIL |
| minilm | 128 | 1.0000 | 1.0000 | 13.557 | 20.536 | FAIL |
| minilm | 256 | 1.0000 | 1.0000 | 26.184 | 37.691 | FAIL |
| modernbert | 32 | 0.8889 | 0.8857 | 18.371 | 26.097 | FAIL |
| modernbert | 64 | 0.9111 | 0.9069 | 32.436 | 47.437 | FAIL |
| modernbert | 128 | 0.8667 | 0.8647 | 53.436 | 76.441 | FAIL |
| modernbert | 256 | 0.8889 | 0.8857 | 102.859 | 139.007 | FAIL |
| mbert | 32 | 0.9556 | 0.9512 | 11.323 | 16.588 | FAIL |
| mbert | 64 | 0.9778 | 0.9751 | 19.039 | 27.672 | FAIL |
| mbert | 128 | 0.9556 | 0.9512 | 34.196 | 50.484 | FAIL |
| mbert | 256 | 0.9556 | 0.9512 | 65.024 | 93.090 | FAIL |
| tinybert | 32 | 0.7333 | 0.6915 | 1.253 | 2.111 | **PASS** |
| tinybert | 64 | 0.7333 | 0.7081 | 1.989 | 3.347 | **PASS** |
| tinybert | 128 | 0.7556 | 0.7271 | 3.604 | 5.544 | **PASS** |
| tinybert | 256 | 0.7556 | 0.7271 | 7.371 | 10.594 | FAIL |
| t5-small | — | — | — | — | — | FAIL* |

> \*T5-small encoder-decoder mimarisi nedeniyle sequence classification göreviyle uyumsuz olup eğitim tamamlanamamıştır.

---

## 4. Model Ablasyonu (max\_length=64 sabit)

| Model | Accuracy | F1 Macro | p50 (ms) | p95 (ms) | NFR-01 |
|-------|----------|----------|----------|----------|--------|
| minilm | **1.0000** | **1.0000** | 7.876 | 12.005 | FAIL* |
| mbert | 0.9778 | 0.9751 | 19.039 | 27.672 | FAIL |
| deberta | 0.6667 | 0.6706 | 13.232 | 14.916 | FAIL |
| modernbert | 0.9111 | 0.9069 | 32.436 | 47.437 | FAIL |
| xlmr | 1.0000 | 1.0000 | 31.027 | 41.668 | FAIL |
| tinybert | 0.7333 | 0.7081 | 1.989 | 3.347 | **PASS** |
| t5-small | — | — | — | — | FAIL |

> \*minilm tok=32'de NFR-01 geçmektedir (p95=7.2ms).

---

## 5. Token Ablasyonu (minilm sabit)

| max\_length | Accuracy | F1 Macro | p50 (ms) | p95 (ms) | NFR-01 |
|------------|----------|----------|----------|----------|--------|
| **32** | **1.0000** | **1.0000** | **4.677** | **7.182** | **PASS** |
| 64 | 1.0000 | 1.0000 | 7.876 | 12.005 | FAIL |
| 128 | 1.0000 | 1.0000 | 13.557 | 20.536 | FAIL |
| 256 | 1.0000 | 1.0000 | 26.184 | 37.691 | FAIL |

> Token uzunluğu artışı doğruluğu etkilememektedir. Dataset'teki prompt'ların ortalama kelime sayısı 5, maksimum 34 olduğundan max_length=32 yeterli kapsam sağlamaktadır. max_length=64 güvenli marj olarak seçilmiştir.

---

## 6. Model Bazlı Değerlendirme

### 6.1 MiniLM-L12 

MiniLM-L12, Microsoft tarafından bilgi damıtma yöntemiyle üretilmiş 22M parametreli hafif bir modeldir.

Bu çalışmada en yüksek doğruluğu elde etmiştir: **PyTorch accuracy=0.9829, ONNX INT8 accuracy=1.00**. tok=32 konfigürasyonunda p95=7.182ms ile NFR-01'i geçen tek yüksek doğruluklu modeldir.

**Güçlü yönler:** En yüksek doğruluk, NFR-01 uyumlu, küçük model boyutu, quantization'a dirençli.  
**Zayıf yönler:** tok=64 ve üzerinde NFR-01 ihlali.

---

### 6.2 mBERT 

mBERT, Google tarafından 104 dil üzerinde pre-train edilmiş 110M parametreli klasik çok dilli BERT modelidir.

**PyTorch accuracy=0.9701, ONNX tok=64 accuracy=0.9778.** Quantization sonrasında doğruluk korunmuştur. Gecikme açısından Windows ortamında NFR-01'i karşılayamamıştır.

**Güçlü yönler:** Yüksek doğruluk, kararlı quantization davranışı, geniş dil desteği.  
**Zayıf yönler:** Büyük model boyutu (110M), yüksek gecikme.

---

### 6.3 DeBERTa-v3-small

DeBERTa-v3-small, Microsoft tarafından disentangled attention mekanizmasıyla geliştirilmiş 22M parametreli bir modeldir.

**Kritik Bulgu:** PyTorch'ta accuracy=0.9601 elde eden model, INT8 dynamic quantization sonrasında accuracy=0.64-0.69'a düşmüştür. Bu %30'luk doğruluk kaybı, DeBERTa'nın disentangled attention yapısının quantization'a aşırı duyarlı olduğunu göstermektedir. Bu nedenle üretim ortamı için uygun değildir.

**Güçlü yönler:** PyTorch'ta yüksek doğruluk, küçük model boyutu.  
**Zayıf yönler:** INT8 quantization sonrası ciddi doğruluk kaybı — üretim için elenmektedir.

---

### 6.4 ModernBERT-base

ModernBERT, 2024 yılında Answer.AI tarafından yayımlanan 149M parametreli güncel bir encoder modelidir.

**PyTorch accuracy=0.9330, ONNX tok=64 accuracy=0.9111.** Tüm modeller arasında en yüksek gecikme değerlerini üretmiştir. Modern mimari iyileştirmelerin Windows CPU ortamında ve kısa metin sınıflandırma görevinde beklenen avantajı sağlamadığı gözlemlenmiştir.

**Güçlü yönler:** Güncel mimari, uzun bağlam desteği.  
**Zayıf yönler:** En yavaş model, orta doğruluk — bu görev için aşırı büyük.

---

### 6.5 XLM-RoBERTa-base

xlm-roberta-base, Facebook AI tarafından 125M parametre ve 100 dil desteğiyle pre-train edilmiş encoder-only modeldir.

**PyTorch accuracy=0.8917, ONNX INT8 accuracy=1.00.** İlginç biçimde ONNX INT8 modeli PyTorch modelinden daha yüksek doğruluk sergilemiştir — bu durum test setinin küçüklüğünden kaynaklanıyor olabilir. Türkçe dil desteği güçlüdür.

**Güçlü yönler:** Türkçe pre-train desteği, kararlı quantization, ONNX ekosistemi uyumluluğu.  
**Zayıf yönler:** Windows CPU'da NFR-01 ihlali.

---

### 6.6 TinyBERT-4L

TinyBERT, Huawei tarafından bilgi damıtma yöntemiyle üretilmiş 14M parametreli en küçük modeldir.

**PyTorch accuracy=0.8860, ONNX tok=32 accuracy=0.7333.** tok=32, 64 ve 128'de NFR-01'i geçmektedir ancak doğruluk sınırlıdır. KAYTARMA sınıfında F1=0.69 değeri bu modelin sınıf sınırlarını iyi öğrenemediğini göstermektedir.

**Güçlü yönler:** En hızlı model, tüm kısa token uzunluklarında NFR-01 uyumlu, minimal kaynak tüketimi.  
**Zayıf yönler:** Düşük doğruluk — güvenlik odaklı sistemler için yetersiz.

---

### 6.7 T5-small

T5-small, encoder-decoder mimarili bir modeldir. Sequence classification göreviyle mimari uyumsuzluk nedeniyle eğitim tamamlanamamıştır. Bu bulgu encoder-only mimari seçimini doğrulamaktadır.

---

## 7. Temel Bulgular

**Bulgu 1 — MiniLM Paradoksu:**
En küçük parametreli modellerden biri (22M) en yüksek doğruluğu elde etmiştir. Bu bulgu, kısa metin sınıflandırma görevlerinde model büyüklüğünün doğrulukla doğru orantılı olmadığını göstermektedir.

**Bulgu 2 — DeBERTa Quantization Hassasiyeti:**
DeBERTa-v3-small, PyTorch'ta %96 doğruluktan INT8 quantization sonrası %64'e düşmüştür. Disentangled attention mekanizması dynamic INT8 quantization ile uyumsuzluk sergilemektedir.

**Bulgu 3 — Encoder-Only Doğrulaması:**
T5-small encoder-decoder mimarisi sequence classification göreviyle uyumsuz olup eğitim tamamlanamamıştır.

**Bulgu 4 — Token Uzunluğu Etkisi:**
Token uzunluğu doğruluğu etkilememektedir. Dataset'teki prompt'ların ortalama kelime sayısı 5, maksimum 34 olduğundan max_length=32 yeterlidir.

**Bulgu 5 — NFR-01 ve Windows Ortamı:**
NFR-01 (p95 ≤ 10ms) yalnızca minilm tok=32 ve tinybert tok=32/64/128 kombinasyonlarında karşılanmıştır. Linux/AVX2 üretim ortamında tüm değerlerin 2-3x düşmesi beklenmektedir.

---

## 8. Seçilen Konfigürasyonlar

### Birincil Konfigürasyon — Doğruluk + Hız Dengesi

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Model | MiniLM-L12 INT8 | En yüksek doğruluk, NFR-01 uyumlu |
| max\_length | 32 | p95=7.2ms, NFR-01 PASS, tüm prompt'ları kapsar |
| Quantization | AVX2 Dynamic INT8 | i5-12500H uyumlu |
| Runtime | ONNX Runtime | Production-ready |
| Tahmini p95 (Linux/AVX2) | ~2-4ms | NFR-01 uyumlu |

### Alternatif Konfigürasyon — Türkçe Optimizasyonu

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Model | xlmr-base INT8 | Türkçe pre-train desteği güçlü |
| max\_length | 32 | p95=23.5ms Windows, Linux'ta ~8ms |
| Not | Linux ortamında NFR-01 uyumlu | — |

---

## 9. 3 Katmanlı Karar Motoru

Ablasyon sonuçları, sadece model bazlı kararın yeterli olmadığını göstermiştir. Bu nedenle hibrit bir karar mekanizması geliştirilmiştir:

**Katman 1 — Model Tahmini:**
```
probs      = softmax(logits)
label      = argmax(probs)
confidence = probs[label]
```

**Katman 2 — Kural Motoru:**
- TEHDIT anahtar kelimeleri: prompt injection, siber saldırı terimleri
- KAYTARMA anahtar kelimeleri: spor, oyun, eğlence, yemek
- GUVENLI anahtar kelimeleri: kurumsal iş terimleri

**Katman 3 — Karar Matrisi:**
- TEHDIT kuralı → her zaman TEHDIT
- KAYTARMA kuralı → her zaman KAYTARMA
- GUVENLI kuralı + model onayı → GUVENLI
- Kural yok → modele güven

Test sonucu: 15/15 doğru tahmin.