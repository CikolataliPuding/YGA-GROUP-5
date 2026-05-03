from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from masking.tr_recognizers import (
    get_tr_phone_recognizer,
    get_tc_kimlik_recognizer,
    get_tr_email_recognizer,
    get_tr_iban_recognizer,
    get_tr_credit_card_recognizer
)

KVKK_ENTITIES = ["PHONE_NUMBER", "TR_ID_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE", "CREDIT_CARD"]

def build_engines():
    # Spacy modelini İngilizce ("en") olarak tanıtıyoruz (KeyError: 'tr' çözümüdür)
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }).create_engine()

    registry = RecognizerRegistry(supported_languages=["en"])
    registry.add_recognizer(get_tr_phone_recognizer())
    registry.add_recognizer(get_tc_kimlik_recognizer())
    registry.add_recognizer(get_tr_email_recognizer())
    registry.add_recognizer(get_tr_iban_recognizer())
    registry.add_recognizer(get_tr_credit_card_recognizer())

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer

def analyze_text(analyzer, text):
    # Dili "en" yaparak çekirdek hatayı engelliyoruz.
    results = analyzer.analyze(text=text, entities=KVKK_ENTITIES, language="en", score_threshold=0.3)
    return results

def anonymize_text(analyzer, anonymizer, text):
    results = analyze_text(analyzer, text)
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized, results