#!/usr/bin/env python3
"""Concurrently check which models actually perform OpenAI-style function
calling, by sending a real `tools` definition and checking the response for
a genuine `tool_calls` entry -- not the `tool_calling` capability flag some
gateways advertise in /v1/models, which is not always accurate.

Env vars:
  OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY

Usage:
  python tool_use_test.py ids.txt results.json

`ids.txt` is a newline-separated list of model ids to test (e.g. the "ok"
ids from smoke_test.py's output) -- deliberately not "every id in the
catalog", since testing tool use on a model that doesn't even answer plain
chat right now is a wasted call.

Prompt-phrasing gotcha (learned the hard way): a hedged instruction like
"...don't guess" made several otherwise tool-capable models ask a
clarifying question instead of calling the tool, even when the answer was
already in the prompt. A blunt imperative naming the exact argument value
fixed most of them -- see PROMPT below.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = os.environ.get("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128")
API_KEY = os.environ["OMNIROUTE_API_KEY"]
TIMEOUT = 45
WORKERS = 8

PROMPT = 'Call the get_weather function with location set to "Paris".'
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name, e.g. Paris"}},
            "required": ["location"],
        },
    },
}]


def call_once(session, model_id):
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "max_tokens": 300,
        "stream": False,
    }
    t0 = time.time()
    try:
        resp = session.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=body, timeout=TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return {"kind": "timeout", "status": None, "elapsed": time.time() - t0}
    except requests.exceptions.RequestException as e:
        return {"kind": "network_error", "status": None, "elapsed": time.time() - t0, "msg": str(e)[:300]}

    elapsed = time.time() - t0
    try:
        data = resp.json()
    except ValueError:
        return {"kind": "bad_json", "status": resp.status_code, "elapsed": elapsed}

    if resp.status_code != 200:
        err = data.get("error")
        msg = (err.get("message") if isinstance(err, dict) else err) or json.dumps(data)[:300]
        return {"kind": "tools_rejected", "status": resp.status_code, "elapsed": elapsed, "msg": msg[:300]}

    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    tool_calls = message.get("tool_calls") or []

    if tool_calls:
        fn = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            loc = (args.get("location") or "") if isinstance(args, dict) else ""
            ok = fn.get("name") == "get_weather" and "paris" in loc.lower()
        except Exception:
            ok = False
        return {"kind": "tool_ok" if ok else "tool_malformed", "status": 200, "elapsed": elapsed,
                "msg": f"name={fn.get('name')} args={str(args_raw)[:120]}"}

    content = (message.get("content") or "").strip()
    return {"kind": "tool_none", "status": 200, "elapsed": elapsed, "msg": content[:200]}


def main():
    ids_path = sys.argv[1] if len(sys.argv) > 1 else "ids.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "toolcall_results.json"

    with open(ids_path, encoding="utf-8") as f:
        model_ids = [line.strip() for line in f if line.strip()]

    session = requests.Session()
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(call_once, session, mid): mid for mid in model_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            model_id = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"kind": "exception", "msg": str(e)[:300]}
            res["id"] = model_id
            results.append(res)
            if i % 25 == 0 or i == len(model_ids):
                print(f"[{i}/{len(model_ids)}] {model_id} -> {res['kind']}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    from collections import Counter
    print(Counter(r["kind"] for r in results), file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
