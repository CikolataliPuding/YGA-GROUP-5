from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Motorları başlatıyoruz
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = "My name is Kinem and my phone number is 0555-555-55-55"

# 1. Veriyi tespit et
results = analyzer.analyze(text=text, entities=None, language='en')

# 2. Veriyi gizle (Maskele)
anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)

print("\n--- SİBER KAYTARMA ÖNLEME: MASKELEME SONUCU ---")
print(anonymized_result.text)