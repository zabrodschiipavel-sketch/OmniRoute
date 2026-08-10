#!/usr/bin/env python3
"""Concurrently probe every model in a catalog with a minimal chat
completion, so you know which ones actually answer right now instead of
trusting the /v1/models listing (a model being *listed* doesn't mean its
provider currently has quota/balance/valid credentials).

Env vars:
  OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY  (see fetch_catalog.py)

Usage:
  python smoke_test.py catalog.json results.json

Skips ids that look like embeddings/audio/image-video-generation/moderation
models -- those need a different endpoint shape (/v1/embeddings, etc.), not
/v1/chat/completions, and would just show up as false failures here.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = os.environ.get("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128")
API_KEY = os.environ["OMNIROUTE_API_KEY"]
TIMEOUT = 45
WORKERS = 8

SKIP_SUBSTRINGS = [
    "embed", "asr", "tts", "whisper", "s2s", "livetranslate",
    "moderation", "content-safety", "tingwu", "realtime",
    "image", "wan2.", "flux.", "veo-free", "veoaifree",
    "seedance", "seedream", "/veo", "computer-use", "captioner",
]

RESET_RE = re.compile(r"reset after (?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?", re.I)


def parse_reset_seconds(msg):
    m = RESET_RE.search(msg or "")
    if not m:
        return None
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def call_once(session, model_id):
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with just the single word: ok"}],
        "max_tokens": 200,
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

    if resp.status_code == 200:
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
        if content:
            return {"kind": "ok", "status": 200, "elapsed": elapsed, "msg": content[:200]}
        if reasoning:
            # spent the whole token budget on chain-of-thought before any
            # visible answer -- still proves the route/model works
            return {"kind": "ok_reasoning_only", "status": 200, "elapsed": elapsed, "msg": reasoning[:200]}
        return {"kind": "empty_content", "status": 200, "elapsed": elapsed}

    err = data.get("error")
    msg = (err.get("message") if isinstance(err, dict) else err) or json.dumps(data)[:300]
    reset_s = parse_reset_seconds(msg)
    if reset_s is not None:
        kind = "quota_short" if reset_s <= 90 else "quota_long"
    elif resp.status_code in (401, 403):
        kind = "auth_or_balance"
    elif resp.status_code == 404:
        kind = "not_found"
    elif resp.status_code == 429:
        kind = "rate_limited"
    else:
        kind = "http_error"
    return {"kind": kind, "status": resp.status_code, "elapsed": elapsed, "msg": msg[:300], "reset_s": reset_s}


def probe(session, model_id):
    r1 = call_once(session, model_id)
    if r1["kind"] == "quota_short" and r1.get("reset_s"):
        # a short, provider-reported cooldown (a few seconds/minutes) is
        # worth one retry; multi-hour/day quotas are not
        time.sleep(min(r1["reset_s"], 90) + 2)
        r2 = call_once(session, model_id)
        r2["retried_after"] = r1["kind"]
        return r2
    return r1


def main():
    catalog_path = sys.argv[1] if len(sys.argv) > 1 else "catalog.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "smoke_results.json"

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    to_test = [m["id"] for m in catalog if not any(s in m["id"].lower() for s in SKIP_SUBSTRINGS)]
    print(f"testing {len(to_test)}/{len(catalog)} models "
          f"(rest need a non-chat endpoint shape)", file=sys.stderr)

    session = requests.Session()
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(probe, session, mid): mid for mid in to_test}
        for i, fut in enumerate(as_completed(futures), 1):
            model_id = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"kind": "exception", "msg": str(e)[:300]}
            res["id"] = model_id
            results.append(res)
            if i % 25 == 0 or i == len(to_test):
                print(f"[{i}/{len(to_test)}] {model_id} -> {res['kind']}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    from collections import Counter
    print(Counter(r["kind"] for r in results), file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
