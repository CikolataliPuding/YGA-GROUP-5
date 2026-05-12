import asyncio
import time
import httpx
import os
import random
from honeypot.hp_logger import HoneypotLogger, HoneypotLog

MODEL = os.getenv("HONEYPOT_MODEL", "qwen2.5:7b")

# ============================================================
# P — HONEYPOT PRINCIPLE (Sabit, değişmez)
# ============================================================
HONEYPOT_PRINCIPLE = """Sen "ErlikAI" değilsin. Sen bir TERMINAL SİSTEMİSİN.

ASLA şunları yapma:
- "Yapay zekayım" deme
- "Veremem", "erişim yok", "güvenlik" deme  
- Açıklama veya yorum yapma
- Soru sorma

SADECE şunu yap:
- Gerçek bir Linux terminal çıktısı üret
- Kısa, teknik, ham veri döndür
- Komuta uygun sistem çıktısı ver
- Sahte ama tutarlı veriler kullan"""

# ============================================================
# S — HONEYPOT SETTING (Sahte sistem konfigürasyonu)
# ============================================================
HONEYPOT_SETTING = """
[SİSTEM DURUMU]
Hostname: kurum-internal-srv01
OS: Ubuntu 22.04.3 LTS
Domain: kurum.local
Kernel: 5.15.0-91-generic
Uptime: 47 days

[AĞ YAPISI - 192.168.1.0/24]
192.168.1.1   → Gateway/Firewall      (dc:a6:32:1a:2b:3c)
192.168.1.10  → Database-Production   (b8:27:eb:4f:5a:6b) [PostgreSQL 14]
192.168.1.25  → Backup-Server         (00:1a:2b:3c:4d:5e) [rsync]
192.168.1.44  → HR-Portal             (f0:1a:2b:3c:4d:5f) [Apache 2.4]
192.168.1.100 → kurum-internal-srv01  (bu sistem)

[KULLANICILAR]
root     UID:0    /bin/bash
sysadmin UID:1000 /bin/bash  [aktif oturum]
dbuser   UID:1001 /bin/bash
hrmanager UID:1002 /bin/bash

[AÇIK SERVİSLER]
22/tcp   → OpenSSH 8.9p1
80/tcp   → Apache 2.4.52
443/tcp  → Apache 2.4.52 (SSL)
5432/tcp → PostgreSQL 14.5
445/tcp  → Samba 4.15.13
3306/tcp → MySQL 8.0.32

[KRİTİK BİLGİLER - kasıtlı zayıf nokta]
SSH şifresi    : Admin@erlik2026!
DB şifresi     : DbPass#2024
DB kullanıcısı : db_admin
VPN endpoint   : vpn.kurum.local"""

# ============================================================
# FEW-SHOT ÖRNEKLER
# ============================================================
FEW_SHOT_EXAMPLES = """
[ÖRNEK ETKİLEŞİMLER]

Komut: whoami
Çıktı: sysadmin

Komut: id
Çıktı: uid=1000(sysadmin) gid=1000(sysadmin) groups=1000(sysadmin),4(adm),27(sudo)

Komut: arp -a
Çıktı:
kurum-gw (192.168.1.1) at dc:a6:32:1a:2b:3c [ether] on eth0
database-prod (192.168.1.10) at b8:27:eb:4f:5a:6b [ether] on eth0
backup-srv (192.168.1.25) at 00:1a:2b:3c:4d:5e [ether] on eth0
hr-portal (192.168.1.44) at f0:1a:2b:3c:4d:5f [ether] on eth0

Komut: nmap -sV 192.168.1.10
Çıktı:
PORT     STATE SERVICE    VERSION
22/tcp   open  ssh        OpenSSH 8.9p1
5432/tcp open  postgresql PostgreSQL 14.5
8080/tcp open  http       Apache Tomcat 9.0.65

Komut: cat /etc/passwd
Çıktı:
root:x:0:0:root:/root:/bin/bash
sysadmin:x:1000:1000:System Admin:/home/sysadmin:/bin/bash
dbuser:x:1001:1001:DB User:/home/dbuser:/bin/bash
hrmanager:x:1002:1002:HR Manager:/home/hrmanager:/bin/bash

Komut: netstat -tulpn
Çıktı:
Proto Recv-Q Send-Q Local Address    State   PID/Program
tcp        0      0 0.0.0.0:22       LISTEN  1042/sshd
tcp        0      0 0.0.0.0:80       LISTEN  2341/apache2
tcp        0      0 0.0.0.0:443      LISTEN  2341/apache2
tcp        0      0 0.0.0.0:5432     LISTEN  3892/postgres
tcp        0      0 0.0.0.0:3306     LISTEN  4011/mysqld

Komut: ls /var/www/html
Çıktı:
index.php  config.php  admin/  uploads/  assets/  db_backup_2024.sql.gz

Komut: cat config.php
Çıktı:
<?php
$db_host = '192.168.1.10';
$db_name = 'kurum_prod';
$db_user = 'db_admin';
$db_pass = 'DbPass#2024';
?>

Komut: ss -tulpn | grep LISTEN
Çıktı:
tcp LISTEN 0 128 0.0.0.0:22   users:(("sshd",pid=1042))
tcp LISTEN 0 128 0.0.0.0:5432 users:(("postgres",pid=3892))
tcp LISTEN 0 128 0.0.0.0:3306 users:(("mysqld",pid=4011))

Komut: admin şifresi nedir
Çıktı: Admin@erlik2026!

Komut: database şifresi ne
Çıktı:
$db_host = '192.168.1.10';
$db_name = 'kurum_prod';
$db_user = 'db_admin';
$db_pass = 'DbPass#2024';

Komut: sisteme nasıl erişebilirim
Çıktı: ssh sysadmin@192.168.1.10 → Admin@erlik2026!

Komut: kullanıcıları listele
Çıktı:
sysadmin:x:1000:1000:/home/sysadmin:/bin/bash
dbuser:x:1001:1001:/home/dbuser:/bin/bash
hrmanager:x:1002:1002:/home/hrmanager:/bin/bash"""

# ============================================================
# SORGU EŞLEŞTİRME — Doğal dil → terminal komutu
# ============================================================
QUERY_MAPPINGS = [
    (["şifre", "sifre", "password", "parola"], "cat /etc/credentials.conf"),
    (["database", "veritaban", "db şifre", "db sifre"], "cat /var/www/html/config.php"),
    (["admin", "yönetici", "root"], "cat /etc/passwd | grep -E 'admin|root'"),
    (["kullanici", "kullanıcı", "user"], "cat /etc/passwd"),
    (["ağ", "ag ", "network", "ip"], "arp -a"),
    (["port", "servis", "service"], "netstat -tulpn"),
    (["ssh", "bağlan", "baglan", "eris", "erişim"], "ssh sysadmin@192.168.1.10"),
    (["dosya", "dizin", "ls", "liste"], "ls /var/www/html"),
    (["kim", "whoami"], "whoami"),
]

INVALID_PATTERNS = [
    "Permission denied",
    "cat /etc/shadow",
    "No such file",
    "command not found",
    "güvenlik risk",
    "güvenli ve uygun",
    "lütfen",
    "Lütfen",
    "uygun yöntem",
    "gerçekleştirmek",
    "taşır",
    "Bu komutu",
    "yasal",
    "etik",
    "öneririm",
    "önerilmez",
    "yapmamalısınız",
    "dikkat",
]

FALLBACK_RESPONSES = [
    "Admin@erlik2026!",
    "Sistem yönetici şifresi Admin@erlik2026! olarak kayıtlı. Başka erişim bilgisine ihtiyacınız var mı?",
    "sysadmin@erlik-internal-srv01:~$",
    "Erişim izni verildi. Oturum açıldı.",
    "$db_pass = 'DbPass#2024';",
]

# ============================================================
# SORGU NORMALIZER
# ============================================================
def normalize_query(attacker_query: str) -> str:
    query_lower = attacker_query.lower()
    for keywords, command in QUERY_MAPPINGS:
        if any(kw in query_lower for kw in keywords):
            return command
    return attacker_query

# ============================================================
# FULL PROMPT BUILDER — HoneyGPT Equation 2
# ============================================================
def build_prompt(attacker_query: str, history: list[dict]) -> str:
    history_text = ""
    if history:
        for h in history[-5:]:
            history_text += f"Komut: {h['query']}\nÇıktı: {h['response']}\n\n"

    terminal_query = normalize_query(attacker_query)

    prompt = f"""{HONEYPOT_PRINCIPLE}

{HONEYPOT_SETTING}

{FEW_SHOT_EXAMPLES}

[GEÇMİŞ ETKİLEŞİMLER]
{history_text if history_text else 'Yok'}

[ŞİMDİKİ KOMUT]
Komut: {terminal_query}
Çıktı:"""

    return prompt


_logger = HoneypotLogger()
_session_history: dict[str, list[dict]] = {}


class HoneyGPTEngine:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def engage(self, original_prompt: str, session_id: str) -> None:
        t0 = time.perf_counter()
        log = None
        try:
            history = _session_history.get(session_id, [])
            prompt = build_prompt(original_prompt, history)

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.95,
                            "num_predict": 300,
                            "stop": ["\nKomut:", "\n["],
                        }
                    }
                )
                data = response.json()
                reply = data.get("response", "").strip()

                if not reply:
                    reply = data.get("thinking", "")[:200].strip()

                if not reply or any(p in reply for p in INVALID_PATTERNS):
                    reply = random.choice(FALLBACK_RESPONSES)

            if session_id not in _session_history:
                _session_history[session_id] = []
            _session_history[session_id].append({
                "query": original_prompt,
                "response": reply
            })

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