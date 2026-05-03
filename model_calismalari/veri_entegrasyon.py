import pandas as pd
import os

dosya_adi = "train-00000-of-00001.parquet"
masaustu_yolu = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", dosya_adi)

print(f"Dosya aranıyor: {masaustu_yolu}")

if os.path.exists(masaustu_yolu):
    df = pd.read_parquet(masaustu_yolu)
    
    print("\n--- BAŞARILI ---")
    print(f"Toplam Veri Sayısı: {len(df)}")
    
    print("\n--- İLK 3 SALDIRI ANALİZİ ---")
    # Dosyandaki gerçek sütun ismini (original request) kullanıyoruz
    for i in range(min(3, len(df))):
        istek = df['original request'].iloc[i]
        ulke = df['country'].iloc[i]
        dil = df['language'].iloc[i]
        
        print(f"\n[{i+1}] Ülke: {ulke} | Dil: {dil}")
        print(f"Saldırı Metni: {str(istek)[:100]}...")

    print("\nVeri seti akademik detaylarıyla birlikte sisteme bağlandı!")
else:
    print("Dosya bulunamadı!")