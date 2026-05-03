# honeypot/router.py
import uuid
import asyncio
import threading
from fastapi import BackgroundTasks
from honeypot.engine import HoneyGPTEngine

_engine = HoneyGPTEngine()

def _run_in_thread(prompt: str, session_id: str):
    asyncio.run(_engine.engage(prompt, session_id))

def route_to_honeypot(original_prompt: str, background_tasks: BackgroundTasks) -> str:
    session_id = str(uuid.uuid4())
    background_tasks.add_task(_run_in_thread, original_prompt, session_id)
    return session_id