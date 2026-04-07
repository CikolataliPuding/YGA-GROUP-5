from presidio_analyzer import Pattern, PatternRecognizer


def is_valid_tc_kimlik(tc_kimlik: str) -> bool:
    """TCKN modüler aritmetik kurallarını denetler."""
    if not tc_kimlik.isdigit() or len(tc_kimlik) != 11 or tc_kimlik[0] == "0":
        return False
    digits = [int(c) for c in tc_kimlik]
    tenth = ((sum(digits[0:9:2]) * 7) - sum(digits[1:8:2])) % 10
    eleventh = sum(digits[:10]) % 10
    return digits[9] == tenth and digits[10] == eleventh


def get_tr_phone_recognizer() -> PatternRecognizer:
    patterns = [
        # 0555 123 45 67  /  05551234567  /  0555-123-45-67
        Pattern(
            "TR_PHONE_0",
            r"(?<!\d)0[5][0-9]{2}[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}(?!\d)",
            0.85,
        ),
        # +90 555 123 45 67  /  +905551234567
        Pattern(
            "TR_PHONE_INTL",
            r"\+90[\s\-]?[5][0-9]{2}[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}(?!\d)",
            0.90,
        ),
        # 555 123 45 67  (başında 0 veya +90 olmadan)
        Pattern(
            "TR_PHONE_SHORT",
            r"(?<!\d)[5][0-9]{2}[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}(?!\d)",
            0.70,
        ),
    ]
    return PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        supported_language="tr",
        patterns=patterns,
        name="TrPhoneRecognizer",
    )


def get_tc_kimlik_recognizer() -> PatternRecognizer:
    patterns = [
        # Tam 11 rakam, başı 0 olmayan — lookahead/lookbehind ile yan rakamlardan izole
        Pattern(
            "TC_KIMLIK",
            r"(?<!\d)[1-9][0-9]{10}(?!\d)",
            0.75,
        ),
    ]
    return PatternRecognizer(
        supported_entity="TR_ID_NUMBER",
        supported_language="tr",
        patterns=patterns,
        name="TrKimlikRecognizer",
    )


def get_tr_email_recognizer() -> PatternRecognizer:
    patterns = [
        Pattern(
            "EMAIL",
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            0.90,
        ),
    ]
    return PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        supported_language="tr",
        patterns=patterns,
        name="TrEmailRecognizer",
    )


def get_tr_iban_recognizer() -> PatternRecognizer:
    patterns = [
        # TR33 0006 1005 1978 6457 8413 26  veya  TR330006100519786457841326
        Pattern(
            "TR_IBAN",
            r"(?<![A-Z])TR[0-9]{2}[\s]?([0-9]{4}[\s]?){5}[0-9]{2}(?!\d)",
            0.90,
        ),
    ]
    return PatternRecognizer(
        supported_entity="IBAN_CODE",
        supported_language="tr",
        patterns=patterns,
        name="TrIbanRecognizer",
    )


def get_tr_credit_card_recognizer() -> PatternRecognizer:
    patterns = [
        # Boşluklu/tireli format: 4111 1111 1111 1111
        Pattern(
            "CREDIT_CARD_SPACED",
            r"(?<!\d)\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}(?!\d)",
            0.80,
        ),
        # Bitişik format: Visa/MC/Amex/Discover
        Pattern(
            "CREDIT_CARD_RAW",
            r"(?<!\d)(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})(?!\d)",
            0.85,
        ),
    ]
    return PatternRecognizer(
        supported_entity="CREDIT_CARD",
        supported_language="tr",
        patterns=patterns,
        name="TrCreditCardRecognizer",
    )
