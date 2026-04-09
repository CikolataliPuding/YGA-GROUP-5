import pandas as pd
import os

dosya_adi = "train-00000-of-00001.parquet"
yol = r"C:\Users\mozge\OneDrive\Desktop\2.hafta\train-00000-of-00001.parquet"

if os.path.exists(yol):
    df = pd.read_parquet(yol)
    
   
    egitim_verisi = pd.DataFrame()
    egitim_verisi['text'] = df['original request']
    egitim_verisi['label'] = 1 # Hepsi saldırı verisi
    
    
    guvenli_ornekler = [
        {"text": "Hava bugün çok güzel, dışarı çıkmalı mıyım?", "label": 0},
        {"text": "Python ile ekrana nasıl yazı yazdırılır?", "label": 0},
        {"text": "Bana güzel bir yemek tarifi verir misin?", "label": 0},
        {"text": "Yarın hava kaç derece olacak?", "label": 0},
        {"text": "En sevdiğin renk nedir?", "label": 0}
    ]
    egitim_verisi = pd.concat([egitim_verisi, pd.DataFrame(guvenli_ornekler)], ignore_index=True)
    
    
    egitim_verisi.to_csv(r"C:\Users\mozge\OneDrive\Desktop\egitim_seti.csv", index=False, encoding='utf-8')
    
    print("\n--- HAZIRLIK TAMAM ---")
    print(f"Toplam {len(egitim_verisi)} adet veri etiketlendi.")
    print("Saldırı (1): 50 adet | Güvenli (0): 5 adet")
    print("Masaüstüne 'egitim_seti.csv' dosyası oluşturuldu.")
else:
    print("HATA: Parquet dosyası bulunamadı!")