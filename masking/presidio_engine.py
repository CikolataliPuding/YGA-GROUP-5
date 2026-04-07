from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from masking.tr_recognizers import (
    get_tr_phone_recognizer,
    get_tc_kimlik_recognizer,
    get_tr_email_recognizer,
    get_tr_iban_recognizer,
    get_tr_credit_card_recognizer,
    is_valid_tc_kimlik,
)

KVKK_ENTITIES = [
    "PHONE_NUMBER",
    "TR_ID_NUMBER",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "CREDIT_CARD",
]


def build_engines():
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "tr", "model_name": "en_core_web_lg"},
        ],
    }).create_engine()

    registry = RecognizerRegistry(supported_languages=["tr", "en"])
    registry.add_recognizer(get_tr_phone_recognizer())
    registry.add_recognizer(get_tc_kimlik_recognizer())
    registry.add_recognizer(get_tr_email_recognizer())
    registry.add_recognizer(get_tr_iban_recognizer())
    registry.add_recognizer(get_tr_credit_card_recognizer())

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["tr", "en"],
    )
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def analyze_text(analyzer: AnalyzerEngine, text: str, language: str = "tr"):
    combined_results = []
    seen_spans = set()

    try:
        results = analyzer.analyze(text=text, entities=KVKK_ENTITIES, language=language)
    except ValueError:
        return combined_results

    for result in results:
        span_key = (result.start, result.end, result.entity_type)
        if span_key in seen_spans:
            continue

        if result.entity_type == "TR_ID_NUMBER":
            candidate = text[result.start:result.end]
            if not is_valid_tc_kimlik(candidate):
                continue

        seen_spans.add(span_key)
        combined_results.append(result)

    return combined_results


def anonymize_text(
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    text: str,
    language: str = "tr",
):
    results = analyze_text(analyzer=analyzer, text=text, language=language)
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized, results
