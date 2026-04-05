import streamlit as st
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Motorları arka planda bir kez çalıştırıyoruz
@st.cache_resource
def load_engines():
    return AnalyzerEngine(), AnonymizerEngine()

analyzer, anonymizer = load_engines()

# Web Sayfası Başlığı
st.title("🛡️ Siber Kaytarma Önleme")
st.subheader("KVKK Maskeleme Arayüzü")

# Kullanıcıdan giriş al
user_input = st.text_area("Mesajınızı buraya yazın:", placeholder="Örn: Benim adım Kinem, numaram 0555...")

if st.button("Verileri Maskele ve Kontrol Et"):
    if user_input:
        # 1. Analiz et
        results = analyzer.analyze(text=user_input, entities=[], language='en')
        
        # 2. Maskele
        anonymized = anonymizer.anonymize(text=user_input, analyzer_results=results)
        
        # Sonuçları ekrana bas
        st.success("İşlem Başarılı!")
        st.write("**Orijinal Metin:**", user_input)
        st.warning(f"**Maskelenmiş Metin:** {anonymized.text}")
        
        if len(results) > 0:
            st.info(f"Toplam {len(results)} adet hassas veri gizlendi.")
    else:
        st.error("Lütfen bir metin girin!")