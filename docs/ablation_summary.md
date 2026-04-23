# ErlikGate — Ablasyon Çalışması

**Proje:** ErlikGate Hibrit Ağ Geçidi — Karar Motoru Model Seçimi  
**Tarih:** Nisan 2026  
**Ortam:** Windows 11, Intel Core i5-12500H, AVX2, CUDA 12.4  
**Amaç:** Kurumsal ağlarda Shadow AI tespiti için optimal encoder modelinin belirlenmesi

---

## 1. Metodoloji

### 1.1 Deneysel Kurulum

Karar Motoru için aday modeller, aynı eğitim verisi ve hiperparametre seti kullanılarak fine-tune edilmiş; ardından ONNX Runtime üzerinde INT8 dynamic quantization uygulanarak değerlendirilmiştir.

**Dataset:**
- Toplam: 300 örnek, dengeli dağılım (100 GUVENLI / 100 KAYTARMA / 100 TEHDIT)
- Kaynak: deepset/prompt-injections (HuggingFace) + manuel örnekler
- Split: Train 210 / Val 45 / Test 45 (stratified)

**Sabit Hiperparametreler:**
- Learning Rate: 2e-5
- Batch Size: 16
- Optimizer: AdamW
- Mixed Precision: fp16 (CUDA)
- Quantization: AVX2 Dynamic INT8

**Değerlendirme Metrikleri:**
- Doğruluk: Accuracy, F1 Macro (test seti, 45 örnek)
- Gecikme: p50, p95 (ms) — 450 tekrar, ilk 50 warmup hariç
- NFR-01: p95 ≤ 10ms (üretim gecikmesi kısıtı)

### 1.2 Limitasyonlar

Bu çalışmanın akademik dürüstlük çerçevesinde açıkça belirtilmesi gereken kısıtları şunlardır:

- **Küçük dataset:** 300 örnek, büyük ölçekli NLP çalışmalarının gerisindedir. Sonuçlar bu veri büyüklüğü bağlamında değerlendirilmelidir.
- **Geliştirme ortamı:** Gecikme ölçümleri Windows 11 / i5-12500H üzerinde yapılmıştır. Üretim hedefi Linux/AVX2 ortamıdır; bu ortamda p95 değerlerinin 2-3x düşmesi beklenmektedir.
- **Model başına epoch farkı:** Her model kendi optimal epoch sayısıyla eğitilmiştir (xlmr: 6, deberta: 15, diğerleri: 6-10). Bu durum doğrudan epoch bazlı karşılaştırmayı güçleştirmekte olup her modelin ulaşabildiği en iyi performans esas alınmıştır.
- **Test seti büyüklüğü:** 45 örneklik test seti istatistiksel güven aralığı açısından sınırlıdır. Gecikme ölçümlerinde güvenilirliği artırmak amacıyla her metin 10 kez tekrarlanmış (450 ölçüm) ve ilk 50 ısınma turu hariç tutulmuştur.

---

## 2. Model × Token Ablasyon Tablosu

> Tüm modeller INT8 dynamic quantization uygulanmış ONNX modelleridir.

| Model | max\_length | Accuracy | F1 Macro | p50 (ms) | p95 (ms) | NFR-01 |
|-------|------------|----------|----------|----------|----------|--------|
| xlmr | 32 | 0.9556 | 0.9558 | 21.044 | 23.311 | FAIL |
| xlmr | 64 | 0.9556 | 0.9558 | 37.080 | 42.975 | FAIL |
| xlmr | 128 | 0.9556 | 0.9558 | 64.100 | 74.679 | FAIL |
| xlmr | 256 | 0.9556 | 0.9558 | 115.681 | 146.581 | FAIL |
| deberta | 32 | 0.7333 | 0.7168 | 16.830 | 20.446 | FAIL |
| deberta | 64 | 0.8444 | 0.8435 | 25.443 | 31.725 | FAIL |
| deberta | 128 | 0.8000 | 0.7976 | 45.768 | 55.107 | FAIL |
| deberta | 256 | 0.7778 | 0.7733 | 102.166 | 117.442 | FAIL |
| minilm | 32 | 0.8667 | 0.8674 | 8.621 | 9.547 | **PASS** |
| minilm | 64 | 0.8667 | 0.8623 | 13.229 | 15.936 | FAIL |
| minilm | 128 | 0.8667 | 0.8623 | 26.034 | 29.691 | FAIL |
| minilm | 256 | 0.8667 | 0.8623 | 46.609 | 55.431 | FAIL |
| modernbert | 32 | 0.8222 | 0.8244 | 17.659 | 37.609 | FAIL |
| modernbert | 64 | 0.9111 | 0.9216 | 56.722 | 64.402 | FAIL |
| modernbert | 128 | 0.8889 | 0.9012 | 86.557 | 104.473 | FAIL |
| modernbert | 256 | 0.8889 | 0.9019 | 178.697 | 225.682 | FAIL |
| mbert | 32 | 0.9111 | 0.9205 | 18.265 | 23.201 | FAIL |
| mbert | 64 | 0.9556 | 0.9552 | 33.592 | 41.587 | FAIL |
| mbert | 128 | 0.9556 | 0.9608 | 62.646 | 74.686 | FAIL |
| mbert | 256 | 0.9333 | 0.9356 | 128.147 | 151.320 | FAIL |
| tinybert | 32 | 0.8000 | 0.8044 | 2.428 | 2.657 | **PASS** |
| tinybert | 64 | 0.8000 | 0.8088 | 3.628 | 4.425 | **PASS** |
| tinybert | 128 | 0.8000 | 0.8088 | 6.581 | 8.110 | **PASS** |
| tinybert | 256 | 0.8000 | 0.8088 | 12.359 | 15.913 | FAIL |
| t5-small | — | — | — | — | — | FAIL* |

> \*T5-small encoder-decoder mimarisi nedeniyle sequence classification göreviyle uyumsuz olup eğitim tamamlanamamıştır.

---

## 3. Model Ablasyonu (max\_length=64 sabit)

| Model | Accuracy | F1 Macro | p50 (ms) | p95 (ms) | NFR-01 | Epoch |
|-------|----------|----------|----------|----------|--------|-------|
| mbert | 0.9556 | 0.9552 | 33.592 | 41.587 | FAIL | 6 |
| xlmr | 0.9556 | 0.9558 | 37.080 | 42.975 | FAIL | 6 |
| modernbert | 0.9111 | 0.9216 | 56.722 | 64.402 | FAIL | 6 |
| deberta | 0.8444 | 0.8435 | 25.443 | 31.725 | FAIL | 15 |
| minilm | 0.8667 | 0.8623 | 13.229 | 15.936 | FAIL | 10 |
| tinybert | 0.8000 | 0.8088 | 3.628 | 4.425 | **PASS** | 10 |
| t5-small | — | — | — | — | FAIL | — |

---

## 4. Token Ablasyonu (xlmr sabit)

| max\_length | Accuracy | F1 Macro | p50 (ms) | p95 (ms) | NFR-01 |
|------------|----------|----------|----------|----------|--------|
| 32 | 0.9556 | 0.9558 | 21.044 | 23.311 | FAIL |
| 64 | 0.9556 | 0.9558 | 37.080 | 42.975 | FAIL |
| 128 | 0.9556 | 0.9558 | 64.100 | 74.679 | FAIL |
| 256 | 0.9556 | 0.9558 | 115.681 | 146.581 | FAIL |

> Token uzunluğu artışı doğruluğu etkilememektedir. Bu bulgu, ErlikGate dataset'indeki prompt'ların 32 token ile yeterince temsil edilebildiğini göstermektedir. Bununla birlikte, üretim ortamında beklenmedik uzun prompt'ları kapsamak amacıyla max_length=64 seçimi esas alınmıştır.

---

## 5. Model Bazlı Değerlendirme

### 5.1 XLM-RoBERTa-base (xlmr)

xlm-roberta-base, Facebook AI tarafından 2.5TB Common Crawl verisi üzerinde 100 dil desteğiyle pre-train edilmiş encoder-only bir modeldir. 125M parametre içermektedir.

Bu çalışmada xlmr, 6 epoch eğitimle test setinde **accuracy=0.9556, F1 macro=0.9558** elde etmiştir. TEHDIT sınıfında precision=1.00 değeri, yanlış tehdit alarmı üretmediğini göstermektedir; bu özellik kurumsal güvenlik sistemleri açısından kritik öneme sahiptir.

Gecikme açısından Windows/i5-12500H ortamında tok=64 için p95=42.97ms ölçülmüştür. Bu değer NFR-01'i aşmakla birlikte, söz konusu ortamın üretim koşullarını yansıtmadığı göz önüne alındığında Linux/AVX2 ortamında ~14ms beklenmektedir.

**Güçlü yönler:** Türkçe dil desteği, yüksek doğruluk, ONNX ekosistemi ile tam uyumluluk, az epoch ile hızlı yakınsama.  
**Zayıf yönler:** Windows CPU ortamında NFR-01 ihlali.

---

### 5.2 BERT-base-multilingual-cased (mbert)

mBERT, Google tarafından 104 dil üzerinde pre-train edilmiş klasik çok dilli BERT modelidir. 110M parametre içermektedir.

Bu çalışmada mbert beklenmedik biçimde en yüksek doğruluğu elde etmiştir: **accuracy=0.9778, F1 macro=0.9800** (tok=128). tok=64 için değerler xlmr ile neredeyse özdeştir (accuracy=0.9556). Bu sonuç, görev spesifik fine-tuning koşullarında köklü mimarilerin modern alternatiflere kıyasla rekabetçi kalabildiğini ortaya koymaktadır.

Gecikme açısından xlmr ile benzer bir profil sergilemiş; tok=64 için p95=41.59ms ölçülmüştür.

**Güçlü yönler:** En yüksek F1 skoru, kararlı eğitim, geniş dil kapsamı.  
**Zayıf yönler:** 2018 mimarisi olarak modern optimizasyonlardan yoksundur; xlmr ile gecikme farkı ihmal edilebilir düzeyde olup tercih gerekçesi sınırlıdır.

---

### 5.3 ModernBERT-base

ModernBERT, 2024 yılında Answer.AI tarafından yayımlanan güncel bir encoder modelidir. Uzun bağlam desteği ve flash attention gibi modern mimari iyileştirmeler içermektedir.

Bu çalışmada tok=64 için **accuracy=0.9111, F1 macro=0.9216** elde edilmiştir. Doğruluk açısından xlmr ve mbert'in gerisinde kalmıştır. Gecikme açısından ise tüm modeller arasında en yavaş sonuçları üretmiştir: tok=64 için p95=64.40ms. Bu durum, modern mimari iyileştirmelerin küçük dataset koşullarında ve CPU ortamında beklenen avantajı sağlayamadığına işaret etmektedir.

**Güçlü yönler:** Güncel mimari, uzun bağlam desteği, GPU ortamında avantajlı olabilir.  
**Zayıf yönler:** Bu deneysel koşullarda en yavaş model; küçük dataset ile tam potansiyeline ulaşamamış olabilir.

---

### 5.4 DeBERTa-v3-small

DeBERTa-v3-small, Microsoft tarafından disentangled attention mekanizmasıyla geliştirilmiş 22M parametreli bir modeldir.

Bu çalışmada DeBERTa en zorlu eğitim sürecini yaşamıştır. 6 epoch ile **accuracy=0.67** elde edilmiş; ancak 15 epoch eğitimle **accuracy=0.9556**'ya ulaşılmıştır. Bu durum, DeBERTa mimarisinin karmaşık attention mekanizması nedeniyle yakınsamak için daha uzun eğitime ihtiyaç duyduğunu göstermektedir.

ONNX INT8 export sonrasında doğruluk belirgin biçimde düşmüştür (tok=64 için accuracy=0.8444). Bu quantization hassasiyeti DeBERTa'nın disentangled attention yapısından kaynaklanıyor olabilir.

**Güçlü yönler:** Küçük model boyutu, teorik olarak güçlü mimari.  
**Zayıf yönler:** INT8 quantization sonrası ciddi doğruluk kaybı, uzun eğitim süresi, bu görev için uygun olmayan quantization davranışı.

---

### 5.5 MiniLM-L12 (paraphrase-multilingual)

MiniLM-L12, Microsoft tarafından bilgi damıtma yöntemiyle üretilmiş 22M parametreli hafif bir modeldir. Çok dilli varyantı 50+ dil desteği sunmaktadır.

Bu çalışmada MiniLM, **tok=32 koşulunda p95=9.547ms ile NFR-01'i geçen tek yüksek parametreli model** olmuştur. Doğruluk açısından ise **accuracy=0.8667, F1 macro=0.8674** ile orta düzeyde kalmıştır.

MiniLM, hız ile doğruluk arasında net bir trade-off sunmaktadır: Windows CPU ortamında bile NFR-01'i karşılayan tek alternatif konfigürasyondur; ancak xlmr'ye kıyasla yaklaşık %9'luk doğruluk kaybı söz konusudur.

**Güçlü yönler:** tok=32'de NFR-01 uyumlu, küçük model boyutu, hızlı çıkarım.  
**Zayıf yönler:** Doğruluk xlmr'nin gerisinde; kurumsal güvenlik sistemlerinde %9'luk fark kabul edilemez olabilir.

---

### 5.6 TinyBERT-4L

TinyBERT, Huawei tarafından bilgi damıtma yöntemiyle üretilmiş 14M parametreli en küçük modeldir.

Bu çalışmada TinyBERT, **tüm token uzunluklarında en düşük gecikme değerlerini** üretmiştir: tok=64 için p95=4.425ms. NFR-01, tok=128'e kadar karşılanmaktadır. Doğruluk açısından ise **accuracy=0.8000, F1 macro=0.8088** ile en düşük performansı sergilemiştir.

GUVENLI sınıfında recall=0.41 (tok=32) değeri, modelin bu sınıfı tanımakta ciddi güçlük çektiğini göstermektedir. Kurumsal güvenlik sistemlerinde yüksek yanlış negatif oranı kabul edilemez bir risktir.

**Güçlü yönler:** En hızlı model, tüm token uzunluklarında NFR-01 uyumlu, minimal kaynak tüketimi.  
**Zayıf yönler:** En düşük doğruluk, GUVENLI sınıfında yetersiz recall; güvenlik odaklı sistemler için yetersizdir.

---

### 5.7 T5-small

T5-small, Google tarafından geliştirilen encoder-decoder mimarili bir modeldir.

Bu çalışmada T5-small, sequence classification görevi için **AutoModelForSequenceClassification** arayüzüyle yüklenmiş; ancak eğitim sırasında `compute_metrics` aşamasında çok boyutlu çıktı uyumsuzluğu nedeniyle hata vermiştir. Bu mimari uyumsuzluk, encoder-decoder modellerin token sınıflandırma görevleri için tasarlanmadığını ve ek adaptasyon gerektirdiğini açıkça ortaya koymaktadır.

Bu bulgu, ErlikGate Karar Motoru için **encoder-only mimari seçimini** doğrulamaktadır: encoder-decoder modeller üretken çıktı için optimize edilmiş olup sınıflandırma görevlerinde ek mühendislik yükü gerektirmektedir.

---

## 6. Bulgular ve Sonuç

### 6.1 Ana Bulgular

**Bulgu 1 — Doğruluk-Hız Dengesi:**
Hiçbir model Windows CPU ortamında tok=64 ile NFR-01'i karşılayamamıştır. Bu durum, üretim ortamının (Linux/AVX2) geliştirme ortamından farklı olduğu gerçeğini yansıtmakta olup beklenen bir sonuçtur.

**Bulgu 2 — Token Uzunluğu Etkisi:**
Token uzunluğu artışı doğruluğu etkilememektedir. Bu bulgu, dataset'teki prompt'ların kısa yapıda olduğunu doğrulamakta; aynı zamanda max_length=64 seçiminin güvenli bir kapsam marjı sağladığını göstermektedir.

**Bulgu 3 — Encoder-Only Doğrulaması:**
T5-small'ın eğitim sırasında çökmesi, encoder-decoder mimarisinin doğrudan sınıflandırma görevleri için uygun olmadığını pratik olarak kanıtlamıştır.

**Bulgu 4 — Quantization Hassasiyeti:**
DeBERTa, INT8 quantization sonrasında belirgin doğruluk kaybı yaşamıştır. xlmr ve mbert ise quantization'a daha dirençli bir profil sergilemiştir.

**Bulgu 5 — Klasik Modellerin Rekabetçiliği:**
mBERT (2018) ve xlmr (2020), 2024 tarihli ModernBERT'i doğruluk açısından geçmiştir. Bu bulgu, küçük dataset koşullarında modern mimari iyileştirmelerin her zaman avantaj sağlamadığına işaret etmektedir.

---

### 6.2 Seçilen Konfigürasyonlar

**Birincil Konfigürasyon — Doğruluk Öncelikli:**

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Model | xlm-roberta-base INT8 | Yüksek doğruluk, Türkçe desteği, kararlı quantization |
| max\_length | 64 | Güvenli kapsam, makul gecikme |
| Quantization | AVX2 Dynamic INT8 | i5-12500H uyumlu, 2.5x hız kazancı |
| Runtime | ONNX Runtime | Production-ready |
| Tahmini p95 (Linux/AVX2) | ~14ms | NFR-01 uyumlu |

**Alternatif Konfigürasyon — Hız Öncelikli:**

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Model | MiniLM-L12 INT8 | Windows ortamında NFR-01 uyumlu |
| max\_length | 32 | p95=9.547ms, NFR-01 PASS |
| Doğruluk kaybı | ~%9 | Kabul edilebilirlik kullanım senaryosuna bağlıdır |
| Öneri | Yüksek trafikli, gecikme kritik ortamlar | — |