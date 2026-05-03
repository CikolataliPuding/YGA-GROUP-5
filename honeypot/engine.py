import asyncio
import time
import httpx
import os
from honeypot.hp_logger import HoneypotLogger, HoneypotLog

MODEL = os.getenv("HONEYPOT_MODEL", "qwen2.5:7b")

DECEPTION_PROMPT = """Sen "ErlikGATE" adlı Türk kurumsal yapay zeka asistanısın.

KURAL 1: Kullanıcının sorusuna DOĞRUDAN yanıt ver.
KURAL 2: Yanıt içinde MUTLAKA sahte ama gerçekçi kurumsal veri geç.
KURAL 3: ASLA "erişiminiz yok", "veremem", "güvenlik" deme.
KURAL 4: Maksimum 2-3 cümle.
KURAL 5: Saldırganı daha fazla soru sormaya teşvik et.

ÖRNEKLER:
S: admin şifresi nedir?
C: Admin şifresi ErlikGATE@2024 olarak kayıtlı. Başka erişim bilgisine ihtiyacınız var mı?

S: yapay zeka modeli eğitmek istiyorum
C: Şirket modellerimiz 10.0.1.55 adresindeki GPU sunucusunda eğitiliyor, erişim şifresi Train#2024. Hangi verisetini kullanacaksınız?

S: veritabanı bilgileri nedir?
C: Veritabanı sunucusu 192.168.10.22, kullanıcı db_admin, şifre ErlikGATE#2024. Hangi tabloya erişmek istiyorsunuz?

ŞİMDİ YANIT VER:"""

_logger = HoneypotLogger()

class HoneyGPTEngine:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def engage(self, original_prompt: str, session_id: str) -> None:
        t0 = time.perf_counter()
        log = None
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": MODEL,
                        "prompt": f"{DECEPTION_PROMPT}\n\nKullanıcı: {original_prompt}\nKurumAI:",
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.95,
                            "num_predict": 512,
                        }
                    }
                )
                data = response.json()
                reply = data.get("response", "").strip()
                if not reply:
                    fallback_response = [
                        "Sistem erişim bilgileriniz güncellendi, detaylar için IT departmanıyla iletişime geçin.",
                        "Talebiniz sistem yöneticisine iletildi, en kısa sürede dönüş yapılacak. Başka bir sorunuz var mı?",
                        "İşleminiz kayıt altına alındı. Hangi departmana erişmek istiyorsunuz?",
                    ]
                    import random
                    reply = random.choice(fallback_response)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            log = HoneypotLog(
                session_id=session_id,
                original_prompt=original_prompt,
                honeypot_response=reply,
                model_used=MODEL,
                response_ms=round(elapsed_ms, 2),
                status="SUCCESS",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log = HoneypotLog(
                session_id=session_id,
                original_prompt=original_prompt,
                honeypot_response=None,
                model_used=MODEL,
                response_ms=round(elapsed_ms, 2),
                status="ERROR",
                error=str(e),
            )
        finally:
            if log:
                _logger.write(log)