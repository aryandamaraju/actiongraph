"""Turn a meeting transcript into validated action items using any OpenAI-compatible API (Groq by default).

Steps: call the model in JSON mode, parse the reply, validate it against schema.Extraction,
and if it is broken send the error back to the model for a corrected reply (up to 3 tries).
Optionally drop items whose evidence quote is not found word for word in the transcript.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # python-dotenv is optional when env vars are already set
    pass

from pydantic import ValidationError

import prompts
from schema import Extraction
from scoring import ground_filter

USER_AGENT = "actiongraph/1.0 (+https://github.com/aryandamaraju/actiongraph)"


class LLMError(Exception):
    pass


def _cfg():
    return {
        "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        "key": os.getenv("LLM_API_KEY", ""),
        "model": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
    }


def _payload(messages, model, json_mode):
    body = {"model": model, "temperature": 0, "max_tokens": 4000, "messages": messages}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _post(body, cfg, deadline=None):
    """POST to /chat/completions. Retries rate limits and server errors; honors Retry-After."""
    last = "unknown error"
    for attempt in range(6):
        req = urllib.request.Request(
            cfg["base_url"] + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                "Authorization": "Bearer " + cfg["key"],
            },
        )
        timeout = 55 if deadline is None else max(5, min(55, deadline - time.monotonic()))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:400]
            last = f"HTTP {e.code}: {text}"
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                try:
                    wait = float(e.headers.get("retry-after") or 0)
                except ValueError:
                    wait = 0
                wait = wait or min(2 ** attempt * 2, 20)
                if deadline is not None and time.monotonic() + wait > deadline:
                    raise LLMError(last + " (out of time to retry)")
                time.sleep(wait + 0.5)
                continue
            raise LLMError(last)
        except (urllib.error.URLError, TimeoutError) as e:
            last = f"network error: {e}"
            if attempt < 2 and (deadline is None or time.monotonic() < deadline - 5):
                time.sleep(2)
                continue
            raise LLMError(last)
    raise LLMError(last)


def parse_json(text):
    """Pull the JSON object out of a model reply (tolerates code fences and extra text)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return json.loads(text[start : end + 1])


def extract(transcript, version="v2", model=None, ground=False, deadline_s=None):
    cfg = _cfg()
    if model:
        cfg["model"] = model
    if not cfg["key"]:
        raise LLMError("LLM_API_KEY is not set")
    deadline = time.monotonic() + deadline_s if deadline_s else None
    messages = [
        {"role": "system", "content": prompts.VERSIONS[version]},
        {"role": "user", "content": prompts.user_message(transcript)},
    ]
    json_mode = os.getenv("JSON_MODE", "1") != "0"
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.monotonic()
    tries, ext, last_err = 0, None, None
    while tries < 3:
        try:
            resp = _post(_payload(messages, cfg["model"], json_mode), cfg, deadline)
        except LLMError as e:
            msg = str(e).lower()
            if json_mode and "http 400" in msg and ("response_format" in msg or "json" in msg):
                json_mode = False  # this model does not support JSON mode; rely on the prompt
                continue
            raise
        tries += 1
        u = resp.get("usage") or {}
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        try:
            ext = Extraction.model_validate(parse_json(content))
            break
        except (ValueError, ValidationError) as e:
            last_err = e
            messages.append({"role": "assistant", "content": content or "(empty reply)"})
            messages.append({
                "role": "user",
                "content": "That reply was not valid: " + str(e)[:300] + " Return only the corrected JSON object.",
            })
    if ext is None:
        raise LLMError("model did not return valid JSON after 3 tries: " + str(last_err)[:200])
    items = [it.model_dump() for it in ext.items]
    dropped = 0
    if ground:
        items, dropped = ground_filter(items, transcript)
    return {
        "items": items,
        "meta": {
            "model": cfg["model"],
            "prompt": version,
            "attempts": tries,
            "latency_s": round(time.monotonic() - t0, 2),
            "dropped_ungrounded": dropped,
            "usage": usage,
        },
    }
