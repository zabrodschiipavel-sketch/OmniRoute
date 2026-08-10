#!/usr/bin/env python3
"""Fetch the full model catalog from a running OmniRoute gateway.

Env vars:
  OMNIROUTE_BASE_URL  gateway base URL, e.g. http://127.0.0.1:20128
  OMNIROUTE_API_KEY   a local gateway API key (Dashboard -> API Keys)

Output: catalog.json -- deduplicated by id (the live endpoint occasionally
repeats a model under the exact same id twice; see README).
"""
import json
import os
import sys
import urllib.request

BASE_URL = os.environ.get("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128")
API_KEY = os.environ["OMNIROUTE_API_KEY"]


def fetch_models():
    req = urllib.request.Request(
        f"{BASE_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["data"]


def main():
    raw = fetch_models()
    seen = set()
    deduped = []
    for m in raw:
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        deduped.append(m)

    out_path = sys.argv[1] if len(sys.argv) > 1 else "catalog.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=1)

    dupes = len(raw) - len(deduped)
    print(f"fetched {len(raw)} entries, {dupes} exact-id duplicate(s), "
          f"wrote {len(deduped)} unique models to {out_path}")


if __name__ == "__main__":
    main()
