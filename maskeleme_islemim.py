import json
import os
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Hız için küçük modeli (sm) en basit yoldan tanıtıyoruz
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}

# Motoru yeni yöntemle başlat
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
anonymizer = AnonymizerEngine()

# --- Geri kalan fonksiyonlar ve dosya yolları aynı kalacak ---

def pii_maskele(metin):
    if not metin: return ""
    results = analyzer.analyze(text=metin, language='en')
    anonymized = anonymizer.anonymize(text=metin, analyzer_results=results)
    return anonymized.text

girdi_dosyasi = "data/raw/erlikgate_v3.jsonl"
cikti_dosyasi = "data/raw/erlikgate_v3_masked.jsonl"

print("🚀 Maskeleme işlemi hızla başlıyor...")

if os.path.exists(girdi_dosyasi):
    with open(girdi_dosyasi, 'r', encoding='utf-8') as f_in, \
         open(cikti_dosyasi, 'w', encoding='utf-8') as f_out:
        
        for satir in f_in:
            veri = json.loads(satir)
            if 'text' in veri:
                veri['text'] = pii_maskele(veri['text'])
            f_out.write(json.dumps(veri, ensure_ascii=False) + '\n')
    
    print(f"✅ İşlem Tamam! Maskelenmiş dosya: {cikti_dosyasi}")
else:
    print(f"❌ Hata: {girdi_dosyasi} bulunamadı!")