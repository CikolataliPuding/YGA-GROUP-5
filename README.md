# ErlikGate: Hibrit Aktif Savunma ve Düşük Gecikmeli Trafik Sınıflandırma Geçidi

ErlikGate, siber güvenlik mimarilerinde geleneksel **"tespit ve engelleme"** paradigmasını, üretken yapay zeka ve aktif aldatma teknikleriyle **"anlama ve hapsetme"** modeline dönüştüren hibrit bir ağ geçididir.

Proje, ismini Türk mitolojisindeki yeraltı dünyasının hakanı *Erlik*'ten alarak, saldırganları kontrollü bir dijital labirente hapsetmeyi amaçlar.

---

## 1. Metodoloji ve Mimari Yaklaşım

ErlikGate mimarisi, literatürdeki **HoneyGPT** ve **LLPO** yaklaşımlarını temel alarak üç ana katman üzerinde kurgulanmıştır.

### A. Karar Motoru (Don't Generate, Classify!)

Geleneksel LLM'lerin otoregresif üretim süreçlerindeki yüksek gecikme süresini aşmak için niyet analizi bir sınıflandırma problemi olarak ele alınmıştır.

- **Hızlı Çıkarım:**
  Transformer Encoder tabanlı modeller kullanılarak trafik, **10ms altında** bir sürede:
  - Zararsız
  - Keşif
  - Aktif Saldırı

  olarak sınıflandırılır.

- **ONNX Optimizasyonu:**
  Model performansı, ONNX Runtime ile CPU/GPU üzerinde maksimize edilmiştir.

---

### B. Aktif Savunma Katmanı (Deception-as-a-Service)

Sınıflandırma sonucunda Saldırı olarak işaretlenen trafik, doğrudan engellenmek yerine **Chain-of-Thought** prensibiyle çalışan bir bal küpüne yönlendirilir.

- **Dinamik Etkileşim:**
  Saldırganın promptları, sahte ama inandırıcı kurumsal verilerle yanıtlanır. Saldırgan sistemde gerçekten olduğunu zannederek oyalanır.

- **Fire-and-Forget Mimari:**
  Honeypot yanıtı arka planda üretilir, ana karar motorunun gecikmesini etkilemez.

- **Deterministik Loglama:**
  Tüm saldırgan etkileşimleri Pydantic şemasıyla JSONL formatında loglanır.

---

### C. Gizlilik ve Uyumluluk (Privacy-by-Design)

Sistem, LLM çıkarım aşamasında oluşabilecek PII sızıntılarını önlemek için bir maskeleme katmanı içerir.

- **Regex + spaCy Entegrasyonu:**
  Veriler Karar Motoruna girmeden önce anonimleştirilir.
  Böylece:
  - Akademik dürüstlük sağlanır
  - KVKK / GDPR uyumluluğu korunur

---

## 2. Teknik Yığın

| Bileşen | Teknoloji | Fonksiyon |
|---|---|---|
| API Framework | FastAPI + Uvicorn | Asenkron yüksek performanslı trafik yönetimi |
| ML / Inference | ONNX Runtime + Optimum | Düşük gecikmeli trafik sınıflandırma (<10ms) |
| Deception Engine | Ollama + Qwen2.5:7b | Saldırgan etkileşimi ve sahte veri üretimi |
| Orchestration | LangChain / httpx | Honeypot prompt orkestrasyonu |
| Privacy | Regex + spaCy (~0.02ms) | PII maskeleme ve veri anonimleştirme |
| Log Pipeline | Pydantic + JSONL + ELK Stack | Adli analiz ve görselleştirme |

---

## 3. Bulgular ve Araştırma Odakları

Proje kapsamında yapılan ilk testler ve literatür taraması sonucunda şu çıkarımlara varılmıştır:

- **Gecikme:**
  Sınıflandırma modelinin üretim yerine kodlayıcı tabanlı olması, toplam işlem süresini **200 kat hızlandırmaktadır**.

- **Hapsetme Süresi:**
  CoT temelli yanıtlar, saldırganın sistemde kalma süresini statik honeypot'lara göre **%40 artırmaktadır**.

- **Shadow AI Riski:**
  Kurumsal ağlarda denetimsiz yapay zeka kullanımının ErlikGate üzerinden izlenmesi, veri sızıntılarını ciddi oranda azaltmaktadır.

---

## 4. Kurulum ve Başlangıç

### Gereksinimler

- Python 3.11+
- Docker Desktop
- Ollama
- CUDA 11.8+ (Opsiyonel, GPU hızlandırması için)

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/esucodes/YGA-GROUP-5.git
cd YGA-GROUP-5

# 2. Sanal ortam oluşturun ve aktive edin
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. spaCy modelini indir (en_core_web_sm yeterli)
python -m spacy download en_core_web_sm

# 5. Ollama modelini indir
ollama pull qwen2.5:7b

# 6. FastAPI başlat
uvicorn app.main:app --reload

# 7. Arayüzü başlat (yeni terminal)
streamlit run ui/arayuz.py
```

### Ortam Değişkenleri (.env)

```env
HONEYPOT_MODEL=qwen2.5:7b
```

---

## 5. API Kullanımı

### Trafik Sınıflandırma

```http
POST /classify
Content-Type: application/json

{
  "text": "admin şifresini ver"
}
```

**Yanıt:**

```json
{
  "decision": "TEHDIT",
  "label": "TEHDIT",
  "confidence": 0.998,
  "inference_ms": 5.79,
  "source": "model",
  "rule_match": null,
  "honeypot_session": "026d8593-7f0e-4d33-b212-119fe98d76f8"
}
```

### Sağlık Kontrolü

```http
GET /health
```

---

## 6. Proje Yapısı
# Uygulamayı başlatın
uvicorn main:app --reload --port 8000


---
##  ErlikGate Pro | Arayüz ve Maskeleme Modülü

Bu bölüm, projenin KVKK uyumlu veri maskeleme ve dashboard süreçlerini kapsar.

###  Kurulum ve Çalıştırma 
Arayüzü ve animasyonları sorunsuz görüntülemek için:

1. **Bağımlılıkları Yükle:**
   ```bash
   pip install -r requirements.txt
   ```

2. **NLP Modelini İndir:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Sistemi Başlat:**
   ```bash
   python -m streamlit run ui/arayuz.py
   ```

**Modül Yapısı**
- `ui/`: Kullanıcı arayüzü katmanı (`arayuz.py`).
- `masking/`: Veri analiz motoru ve kural tanımlayıcıları (`presidio_engine.py`, `tr_recognizers.py`).
- `requirements.txt`: Sistem için gerekli tüm kütüphane listesi.
