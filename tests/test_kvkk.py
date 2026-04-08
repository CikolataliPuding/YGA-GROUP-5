import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from masking.presidio_engine import analyze_text, build_engines
from masking.tr_recognizers import is_valid_tc_kimlik

analyzer, anonymizer = build_engines()

def test_telefon_tr():
    text = "Numaram 0555-123-45-67"
    results = analyze_text(analyzer=analyzer, text=text, language='en')
    assert any(r.entity_type == "PHONE_NUMBER" for r in results), "TR telefon yakalanamadı!"

def test_tc_kimlik():
    text = "TC kimliğim 10000000146"
    results = analyze_text(analyzer=analyzer, text=text, language='en')
    assert any(r.entity_type == "TR_ID_NUMBER" for r in results), "TC kimlik yakalanamadı!"
    assert is_valid_tc_kimlik("10000000146") is True


def test_gecersiz_tc_kimlik():
    text = "TC kimliğim 12345678901"
    results = analyze_text(analyzer=analyzer, text=text, language='en')
    assert not any(r.entity_type == "TR_ID_NUMBER" for r in results), "Geçersiz TC kimlik yanlışlıkla kabul edildi!"

def test_email():
    text = "Mail adresim egemen@example.com"
    results = analyze_text(analyzer=analyzer, text=text, language='en')
    assert any(r.entity_type == "EMAIL_ADDRESS" for r in results), "Email yakalanamadı!"

if __name__ == "__main__":
    for fn_name, fn in [
        ("TR Telefon", test_telefon_tr),
        ("TC Kimlik",  test_tc_kimlik),
        ("Email",      test_email),
    ]:
        try:
            fn()
            print(f"✓ {fn_name} testi geçti")
        except AssertionError as e:
            print(f"✗ {fn_name} testi BAŞARISIZ: {e}")