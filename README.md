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
  Saldırganın komutları sistemin niyetine göre manipüle edilen gerçekçi bir terminal ortamında simüle edilir.

- **Niyet Analizi:**  
  Saldırganın sadece ne yaptığı değil, **neden yaptığı** derinlemesine analiz edilir.

---

### C. Gizlilik ve Uyumluluk (Privacy-by-Design)

Sistem, LLM çıkarım aşamasında oluşabilecek PII sızıntılarını önlemek için bir maskeleme katmanı içerir.

- **Presidio Entegrasyonu:**  
  Veriler Karar Motoruna girmeden önce anonimleştirilir.  
  Böylece:
  - Akademik dürüstlük sağlanır
  - KVKK / GDPR uyumluluğu korunur

---

## 2. Teknik Yığın

| Bileşen            | Teknoloji                      | Fonksiyon                                  |
|------------------|------------------------------|--------------------------------------------|
| API Framework     | FastAPI                       | Asenkron yüksek performanslı trafik yönetimi |
| ML / Inference    | ONNX Runtime / PyTorch        | Düşük gecikmeli trafik sınıflandırma        |
| Deception Logic   | OpenAI / Llama 3   | Saldırgan etkileşimi ve niyet analizi       |
| Privacy           | Microsoft Presidio            | PII maskeleme ve veri anonimleştirme        |
| Cache & State     | Redis                         | Oturum yönetimi ve mükerrer saldırı analizi |
| Log Pipeline      | ELK Stack         | Adli analiz ve görselleştirme               |

---

## 3. Bulgular ve Araştırma Odakları

Proje kapsamında yapılan ilk testler ve literatür taraması sonucunda şu çıkarımlara varılmıştır:

- **Gecikme:**  
  Sınıflandırma modelinin üretim yerine kodlayıcı tabanlı olması, toplam işlem süresini **200 kat hızlandırmaktadır**.

- **Hapsetme Süresi:**  
  CoT temelli yanıtlar, saldırganın sistemde kalma süresini statik honeypot’lara göre **%40 artırmaktadır**.

- **Shadow AI Riski:**  
  Kurumsal ağlarda denetimsiz yapay zeka kullanımının ErlikGate üzerinden izlenmesi, veri sızıntılarını ciddi oranda azaltmaktadır.

---

## 4. Kurulum ve Başlangıç

### Gereksinimler

- Python 3.9+
- Docker
- CUDA 11.8+ (Opsiyonel)

### Adımlar

```bash
# Depoyu klonlayın
git clone https://github.com/egemen/erlikgate.git
cd erlikgate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Gerekli spaCy modelini indirin
python -m spacy download en_core_web_lg
# Çevre değişkenlerini ayarlayın
cp .env.example .env

# Uygulamayı başlatın
uvicorn main:app --reload --port 8000