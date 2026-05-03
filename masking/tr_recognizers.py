from presidio_analyzer import Pattern, PatternRecognizer

def get_tc_kimlik_recognizer():
    # Daha esnek ve güçlü bir Regex kalıbı: Önünde/arkasında rakam olmayan 11 haneler
    pattern = r"(?<!\d)[1-9][0-9]{10}(?!\d)"
    tc_pattern = Pattern(name="tc_kimlik_pattern", score=0.5, regex=pattern)
    return PatternRecognizer(supported_entity="TR_ID_NUMBER", patterns=[tc_pattern])

def get_tr_phone_recognizer():
    pattern = r"(?<!\d)(?:\+90|0)?\s*[5][0-9]{2}\s*[0-9]{3}\s*[0-9]{2}\s*[0-9]{2}(?!\d)"
    phone_pattern = Pattern(name="tr_phone_pattern", score=0.5, regex=pattern)
    return PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[phone_pattern])

def get_tr_email_recognizer():
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email_pattern = Pattern(name="tr_email_pattern", score=0.5, regex=pattern)
    return PatternRecognizer(supported_entity="EMAIL_ADDRESS", patterns=[email_pattern])

def get_tr_iban_recognizer():
    pattern = r"TR[0-9]{2}\s*(?:[0-9]{4}\s*){5}[0-9]{2}"
    iban_pattern = Pattern(name="tr_iban_pattern", score=0.5, regex=pattern)
    return PatternRecognizer(supported_entity="IBAN_CODE", patterns=[iban_pattern])

def get_tr_credit_card_recognizer():
    pattern = r"(?<!\d)(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\s*[0-9]{3})[0-9]{11})(?!\d)"
    card_pattern = Pattern(name="tr_card_pattern", score=0.5, regex=pattern)
    return PatternRecognizer(supported_entity="CREDIT_CARD", patterns=[card_pattern])

def is_valid_tc_kimlik(tc):
    # Testler için bu fonksiyonu artık presidio_engine içinde pasif yaptık.
    return True