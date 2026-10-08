"""OpenRouter client: chat() and embed() over plain HTTP, with retries and a cost log."""
import json
import os
import time

import requests
from dotenv import load_dotenv

from rag import config

load_dotenv(config.ROOT / ".env")
URL = "https://openrouter.ai/api/v1"
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def _headers():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; copy .env.example to .env")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "tokenizer-lit-rag"}


def _post(path, body, timeout=180):
    for attempt in range(6):
        r = requests.post(URL + path, headers=_headers(), json=body, timeout=timeout)
        if r.ok:
            data = r.json()
            if "error" not in data:
                return data
        if attempt == 5 or (not r.ok and r.status_code not in RETRY_STATUSES):
            raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")
        time.sleep(2 ** (attempt + 1))  # 2, 4, 8, 16, 32 s


def _log(kind, model, usage):
    config.LOGS.mkdir(exist_ok=True)
    row = {"ts": round(time.time()), "kind": kind, "model": model,
           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
           "cost": usage.get("cost")}
    with open(config.LOGS / "llm_calls.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def chat(messages, model, schema=None, temperature=0.0, max_tokens=1024):
    """Return (text, cost_usd). With `schema` (a JSON schema dict) the reply is strict JSON."""
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "usage": {"include": True}}
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "out", "strict": True, "schema": schema}}
        body["provider"] = {"require_parameters": True}
    data = _post("/chat/completions", body)
    usage = data.get("usage") or {}
    _log("chat", model, usage)
    return data["choices"][0]["message"]["content"], float(usage.get("cost") or 0.0)


def embed(texts, model, provider=None):
    """Return (list of float vectors in input order, cost_usd)."""
    body = {"model": model, "input": list(texts), "usage": {"include": True}}
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    data = _post("/embeddings", body)
    usage = data.get("usage") or {}
    _log("embed", model, usage)
    vectors = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    return vectors, float(usage.get("cost") or 0.0)


def parse_json(text):
    """Parse a JSON reply, tolerating ```json fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(t)
