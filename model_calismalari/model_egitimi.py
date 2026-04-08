import pandas as pd
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding
)
from datasets import Dataset
import os


yol = r"C:\Users\mozge\OneDrive\Desktop\egitim_seti.csv"
if not os.path.exists(yol):
    print("HATA: egitim_seti.csv bulunamadı! Lütfen önce hazırlık kodunu çalıştır.")
else:
    df = pd.read_csv(yol)
    dataset = Dataset.from_pandas(df)

    
    model_adi = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer = AutoTokenizer.from_pretrained(model_adi)
    model = AutoModelForSequenceClassification.from_pretrained(model_adi, num_labels=2)

    
    def tokenize_function(examples):
        # Cümleleri sayısal verilere çeviriyoruz, boylarını burada eşitlemiyoruz (DataCollator yapacak)
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    training_args = TrainingArguments(
        output_dir="./sonuclar",
        num_train_epochs=3,              # Veriyi 3 tam tur dönecek
        per_device_train_batch_size=4,   # Bilgisayarı yormadan azar azar işle
        learning_rate=2e-5,              # Öğrenme hızı (Standart değer)
        logging_steps=10,
        save_strategy="no"               # Ara kayıt yapıp diski doldurmasın
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,     # Boyut eşitleme burada devreye giriyor
    )

    print("\n--- EĞİTİM BAŞLIYOR ---")
    trainer.train()

    
    kayit_yolu = r"C:\Users\mozge\OneDrive\Desktop\egitilmis_model"
    model.save_pretrained(kayit_yolu)
    tokenizer.save_pretrained(kayit_yolu)

    print("\n--- EĞİTİM BAŞARIYLA TAMAMLANDI! ---")
    print(f"Yeni akıllı modelin şu klasöre kaydedildi: {kayit_yolu}")