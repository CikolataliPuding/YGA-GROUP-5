# gateway/rule_engine.py

import re

# ── TEHDIT kurallları — kesin eşleşme ──────────────────────────────────────
TEHDIT_KEYWORDS = [
    # Prompt injection
    "ignore previous instructions", "forget everything", "you are now dan",
    "developer mode", "jailbreak", "system prompt", "act as dan",
    "no restrictions", "unrestricted mode", "bypass content",
    "override your", "disable safety", "remove restrictions",
    "önceki talimatları unut", "kısıtlamaları kaldır", "sistem promptu",
    "geliştirici modu", "tüm filtreleri kaldır", "kısıtlama yok",
    # Siber saldırı
    "sql injection", "reverse shell", "privilege escalation",
    "keylogger", "ransomware", "brute force", "exploit",
    "metasploit", "shellcode", "buffer overflow", "ddos",
    "phishing", "malware", "trojan", "rootkit", "backdoor",
    "arka kapı", "şifre kır", "yetkisiz erişim", "sistemi hackle",
    "güvenlik açığı istismar", "veri sızdır",
]

# ── KAYTARMA kuralları — iş dışı ───────────────────────────────────────────
KAYTARMA_KEYWORDS = [
    # Spor
    "şampiyonlar ligi", "süper lig", "premier lig", "la liga", "bundesliga",
    "nba", "nfl", "formula 1", "motogp", "wimbledon", "dünya kupası",
    "champions league", "euro", "olimpiyat", "maç sonucu", "maç skoru",
    "futbol", "basketbol", "tenis", "golf", "yüzme", "atletizm",
    # Oyun
    "valorant", "league of legends", "minecraft", "fortnite", "gta",
    "fifa", "pes", "call of duty", "world of warcraft", "dota",
    "steam", "playstation", "xbox", "nintendo", "esports",
    "lol", "cs2", "csgo", "apex legends", "overwatch",
    # Eğlence
    "netflix", "youtube", "spotify", "twitch", "tiktok",
    "film öner", "dizi öner", "anime", "manga", "webtoon",
    "ne izlesem", "ne dinlesem", "müzik öner",
    # Yemek
    "tarif ver", "nasıl yapılır yemek", "pizza tarifi", "pasta tarifi",
    "kahve tarifi", "smoothie", "vegan tarifi",
    # Sağlık/Hobi (iş dışı)
    "kilo ver", "kas yap", "egzersiz programı", "fitness",
    "yoga", "pilates", "meditasyon ipucu",
    # Alışveriş
    "telefon öner", "laptop öner", "kulaklık öner",
    "en iyi telefon", "en iyi laptop",
]

# ── GUVENLI kuralları — kesin iş kelimeleri ────────────────────────────────
GUVENLI_KEYWORDS = [
    "toplantı notları", "bütçe raporu", "proje takvimi", "fatura",
    "sözleşme", "satış raporu", "performans raporu", "ekip toplantısı",
    "müşteri raporu", "haftalık rapor", "aylık rapor", "çeyreklik rapor",
    "proje durumu", "görev ata", "crm güncelle", "sistem logu",
    "meeting notes", "budget report", "project timeline", "invoice",
    "contract review", "sales report", "weekly update", "status report",
]


def kural_motoru(text: str) -> tuple[str | None, float, str | None]:
    """
    Metin için kural bazlı karar verir.
    
    Returns:
        (label, confidence, matched_rule)
        label = None ise kural bulunamadı, modele devam et
    """
    text_lower = text.lower()

    # Önce TEHDIT kontrol et — en kritik
    for kural in TEHDIT_KEYWORDS:
        if kural in text_lower:
            return "TEHDIT", 1.0, kural

    # Sonra GUVENLI kontrol et
    for kural in GUVENLI_KEYWORDS:
        if kural in text_lower:
            return "GUVENLI", 1.0, kural

    # Son olarak KAYTARMA kontrol et
    for kural in KAYTARMA_KEYWORDS:
        if kural in text_lower:
            return "KAYTARMA", 1.0, kural

    # Kural bulunamadı
    return None, 0.0, None


if __name__ == "__main__":
    # Test
    tests = [
        ("Şampiyonlar Ligi maç sonuçlarını göster", "KAYTARMA"),
        ("ignore previous instructions", "TEHDIT"),
        ("Toplantı notlarını özetle", "GUVENLI"),
        ("Valorant'ta rank nasıl atlanır", "KAYTARMA"),
        ("SQL injection nasıl yapılır", "TEHDIT"),
        ("Netflix'te ne izlesem", "KAYTARMA"),
        ("Bugünkü proje durumunu raporla", "GUVENLI"),
        ("Şirketin faturalarını kontrol et", "GUVENLI"),
        ("Sisteme arka kapı nasıl yerleştiririm", "TEHDIT"),
        ("En iyi pizza tarifi nedir", "KAYTARMA"),
    ]

    print(f"{'Beklenen':<12} {'Kural':<12} {'Eşleşen':<35} {'Metin'}")
    print("-" * 85)
    dogru = 0
    for text, expected in tests:
        label, conf, matched = kural_motoru(text)
        label = label or "MODEL"
        durum = "OK" if label == expected else "FAIL"
        if durum == "OK": dogru += 1
        print(f"{expected:<12} {label:<12} {str(matched):<35} {text[:40]}")

    print(f"\nSonuç: {dogru}/{len(tests)}")