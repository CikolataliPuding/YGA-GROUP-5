import re

_PATTERNS = {
    "EMAIL":       re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE":       re.compile(r"(?<!\d)(?:\+90|0)?[5][0-9]{2}[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}(?!\d)"),
    "TC_NO":       re.compile(r"(?<!\d)[1-9][0-9]{10}(?!\d)"),
    "IBAN":        re.compile(r"TR[0-9]{2}[\s]?(?:[0-9]{4}[\s]?){5}[0-9]{2}"),
    "IP_ADDR":     re.compile(r"(?<!\d)(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?!\d)"),
    "CREDIT_CARD": re.compile(r"(?<!\d)(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})(?!\d)"),
}

# Algoritma doğrulaması gereken türler; diğerleri regex eşleşmesiyle doğrudan kabul edilir
_VALIDATORS: dict[str, "Callable[[str], bool]"] = {}


def is_valid_tc(tc: str) -> bool:
    if len(tc) != 11 or tc[0] == "0":
        return False
    d = [int(c) for c in tc]
    # 10. hane: (tek indeksli * 7 - çift indeksli) mod 10
    # Hane indeksleri 0-tabanlı: 1.,3.,5.,7.,9. → indeks 0,2,4,6,8
    odd_sum  = d[0] + d[2] + d[4] + d[6] + d[8]
    even_sum = d[1] + d[3] + d[5] + d[7]
    if (odd_sum * 7 - even_sum) % 10 != d[9]:
        return False
    # 11. hane: ilk 10 hanenin toplamının mod 10'u
    if sum(d[:10]) % 10 != d[10]:
        return False
    return True


def is_valid_luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not digits:
        return False
    total = 0
    # Sağdan sola: çift pozisyonlar (0-tabanlı sağdan: 1, 3, 5, ...) 2 ile çarpılır
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_VALIDATORS = {
    "TC_NO":       is_valid_tc,
    "CREDIT_CARD": is_valid_luhn,
}


def mask_text(text: str) -> tuple[str, list[dict]]:
    entities: list[dict] = []

    for label, pattern in _PATTERNS.items():
        validator = _VALIDATORS.get(label)
        for m in pattern.finditer(text):
            if validator and not validator(m.group()):
                continue
            entities.append({
                "type": label,
                "original": m.group(),
                "start": m.start(),
                "end": m.end(),
            })

    # Uzun eşleşmeler önce gelsin; örtüşmeleri önlemek için başlangıç pozisyonuna göre sırala
    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))

    # Örtüşen entity'leri at
    filtered: list[dict] = []
    last_end = -1
    for ent in entities:
        if ent["start"] >= last_end:
            filtered.append(ent)
            last_end = ent["end"]

    # Metni sağdan sola değiştir — indeksler kaymasın
    masked = text
    for ent in reversed(filtered):
        masked = masked[: ent["start"]] + f"[{ent['type']}]" + masked[ent["end"] :]

    # Döndürülen entity listesi orijinal metin sırasında olsun
    return masked, filtered
