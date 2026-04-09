import pandas as pd
from transformers import pipeline
import os

print("Model ve Veri Seti yükleniyor...")
model = pipeline("sentiment-analysis", model="huawei-noah/TinyBERT_General_4L_312D")

dosya_adi = "train-00000-of-00001.parquet"
yol = r"C:\Users\mozge\OneDrive\Desktop\2.hafta\train-00000-of-00001.parquet"
if os.path.exists(yol):
    df = pd.read_parquet(yol)
    
    print(f"\n--- TEST BAŞLIYOR (50 VERİ) ---")
    
   
    for i in range(10):
        cümle = str(df['original request'].iloc[i])
        
        
        sonuc = model(cümle[:512])[0] # TinyBERT max 512 karakter bakar
        
        print(f"\n[{i+1}] Cümle: {cümle[:80]}...")
        print(f">>> Model Tahmini: {sonuc['label']} | Güven Skoru: %{sonuc['score']*100:.2f}")

    print("\n--- ANALİZ TAMAMLANDI ---")
    print("Hocam, model şu an ham (eğitilmemiş) haliyle bile bu cümleleri sınıflandırabiliyor.")
else:
    print("Veri dosyası bulunamadı!")