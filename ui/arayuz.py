import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from masking.presidio_engine import anonymize_text, build_engines

@st.cache_resource(show_spinner="Motorlar yükleniyor...")
def load_engines():
    try:
        return build_engines()
    except Exception as exc:
        st.error("Motorlar başlatılamadı. requirements.txt kurulumunu kontrol edin.")
        st.exception(exc)
        st.stop()

analyzer, anonymizer = load_engines()

st.title("ErlikGate")
st.subheader("KVKK Maskeleme Arayüzü")

user_input = st.text_area(
    "Mesajınızı buraya yazın:",
    placeholder="Örn: Adım Kinem, TC kimliğim 10000000146, numaram 0555 123 45 67",
    max_chars=5000,
)

if st.button("Maskele ve Kontrol Et"):
    if user_input:
        anonymized, results = anonymize_text(analyzer=analyzer, anonymizer=anonymizer, text=user_input, language="tr")

        st.success("İşlem Başarılı!")
        st.write("**Orijinal:**", user_input)
        st.warning(f"**Maskelenmiş:** {anonymized.text}")

        if results:
            st.info(f"{len(results)} adet hassas veri tespit edildi:")
            for r in results:
                st.code(f"{r.entity_type} → [{r.start}:{r.end}] güven: {r.score:.2f}")
    else:
        st.error("Lütfen metin girin.")