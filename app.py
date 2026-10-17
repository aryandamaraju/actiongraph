import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

import extract

app = FastAPI(title="ActionGraph")

MAX_CHARS = 12000
MIN_CHARS = 20
WINDOW_S, MAX_PER_WINDOW = 600, 10
_hits = defaultdict(deque)  # per-instance, best-effort limit to protect the free API quota


class Meeting(BaseModel):
    transcript: str


def _client_ip(request: Request):
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _rate_limit(ip):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > WINDOW_S:
        q.popleft()
    if len(q) >= MAX_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many requests from this address. Try again in a few minutes.")
    q.append(now)


@app.get("/health")
def health():
    cfg = extract._cfg()
    return {"status": "ok", "model": cfg["model"], "prompt": os.getenv("PROMPT_VERSION", "v3"), "key_set": bool(cfg["key"])}


@app.post("/extract")
def extract_items(m: Meeting, request: Request):
    text = m.transcript.strip()
    if len(text) < MIN_CHARS:
        raise HTTPException(status_code=400, detail="Paste a longer transcript.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Transcript is too long (max {MAX_CHARS} characters).")
    _rate_limit(_client_ip(request))
    try:
        out = extract.extract(text, version=os.getenv("PROMPT_VERSION", "v3"), ground=True, deadline_s=50)
    except Exception as e:
        print("extract error:", type(e).__name__, str(e)[:200])
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {str(e)[:160]}")
    return out


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")
