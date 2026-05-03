import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

LOG_PATH = Path("logs/honeypot.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)

class HoneypotLog(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str
    original_prompt: str
    honeypot_response: Optional[str] = None
    model_used: str
    response_ms: float
    status: str
    error: Optional[str] = None
    threat_type: str = "TEHDIT"
    source: str = "honeygpt"

class HoneypotLogger:
    def write(self, log: HoneypotLog) -> None:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(log.model_dump_json() + "\n")