#!/usr/bin/env python3
"""Match an OmniRoute catalog (fetch_catalog.py output) against Artificial
Analysis's Intelligence Index, so every model id gets a comparable quality
score instead of just a name.

Env vars:
  AA_API_KEY   Artificial Analysis Data API key (artificialanalysis.ai)

Usage:
  python match_aa.py catalog.json aa_matches.json

Matching is id-normalization + numeric-token-gated fuzzy matching, not a
plain string-similarity threshold -- see the NUMERIC GATE section for why
that matters (it is the single biggest source of wrong matches otherwise).
"""
import json
import os
import re
import sys
import difflib
import urllib.request
from collections import defaultdict


# ---------------------------------------------------------------- fetch AA

def fetch_aa_catalog():
    """Pull the full free-tier language-model catalog (paginated)."""
    key = os.environ["AA_API_KEY"]
    models, page = [], 1
    while True:
        req = urllib.request.Request(
            f"https://artificialanalysis.ai/api/v2/language/models/free?page={page}",
            headers={"x-api-key": key},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        models.extend(data["data"])
        if not data["pagination"]["has_more"]:
            break
        page += 1
    return models


# ---------------------------------------------------------- normalization

DATE_RE = re.compile(r"-(20\d{2}-\d{2}-\d{2}|20\d{6})$")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
PAREN_RE = re.compile(r"\([^)]*\)")

# Cloudflare Workers AI ids look like "@cf/org/model" and OmniRoute prefixes
# every id with its own provider slug ("ali/", "dgrid/openai/", ...) -- we
# only care about the last path segment plus a creator guess from the path.
ORG_PATH_CREATOR = {
    "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
    "deepseek": "DeepSeek", "deepseek-ai": "DeepSeek",
    "x-ai": "SpaceXAI", "z-ai": "Z AI", "zai-org": "Z AI",
    "qwen": "Alibaba", "alibaba": "Alibaba",
    "minimax": "MiniMax", "moonshotai": "Kimi",
    "meta": "Meta", "meta-llama": "Meta",
    "xiaomi": "Xiaomi", "mistralai": "Mistral", "mistral": "Mistral",
    "nvidia": "NVIDIA", "ibm-granite": "IBM",
}
NAME_CREATOR_PATTERNS = [
    (re.compile(r"^claude"), "Anthropic"), (re.compile(r"^(gpt|o[0-9]|codex)"), "OpenAI"),
    (re.compile(r"^(gemini|gemma)"), "Google"), (re.compile(r"^llama"), "Meta"),
    (re.compile(r"^mistral"), "Mistral"), (re.compile(r"^(glm|zai)"), "Z AI"),
    (re.compile(r"^kimi"), "Kimi"), (re.compile(r"^minimax"), "MiniMax"),
    (re.compile(r"^deepseek"), "DeepSeek"), (re.compile(r"^nemotron"), "NVIDIA"),
    (re.compile(r"^(qwen|qwq|qvq)"), "Alibaba"), (re.compile(r"^grok"), "SpaceXAI"),
]


def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def name_base(name):
    return PAREN_RE.sub("", name).strip()


def normalize_id(model_id):
    """OmniRoute id -> (tail, creator hint). `tail` is the last path
    segment with the date/colon-tag stripped; NOT with -preview/-latest
    stripped -- do that later, and only as a last resort (see below)."""
    suffix = model_id.split("/", 1)[1] if "/" in model_id else model_id
    parts = suffix.split("/")
    tail = parts[-1].split(":")[0]
    tail = DATE_RE.sub("", tail)

    creator = None
    if len(parts) > 1 and parts[0] in ORG_PATH_CREATOR:
        creator = ORG_PATH_CREATOR[parts[0]]
    elif parts[0] == "@cf" and len(parts) > 2 and parts[1] in ORG_PATH_CREATOR:
        creator = ORG_PATH_CREATOR[parts[1]]
    if not creator:
        for rx, c in NAME_CREATOR_PATTERNS:
            if rx.match(slugify(tail)):
                creator = c
                break
    return suffix, tail, creator


NOISE_TAIL_SUFFIXES = ["-instruct", "-it", "-chat", "-lora", "-fp8", "-versatile"]


def candidate_slug_variants(tail):
    """Ordered, most-faithful-first. Order matters: an early exact hit
    always wins over a later, more-stripped one, so a real AA entry named
    "... (Preview)" is found before we ever consider dropping "-preview"
    (some models have BOTH a base and a Preview entry with different
    scores -- e.g. Qwen3 Max 24.5 vs Qwen3 Max (Preview) 19.4; stripping
    "-preview" unconditionally silently picked the wrong one)."""
    t = re.sub(r"^@?cf/", "", tail)
    variants = []

    def add(v):
        if v and v not in variants:
            variants.append(v)

    add(t)
    if "/" in t:
        add(t.split("/")[-1])
    add(re.sub(r"-free$", "", t))
    stripped = t
    changed = True
    while changed:
        changed = False
        for suf in NOISE_TAIL_SUFFIXES:
            if stripped.endswith(suf):
                stripped = stripped[: -len(suf)]
                add(stripped)
                changed = True
    for suf in ("-preview", "-latest"):
        if t.endswith(suf):
            add(t[: -len(suf)])
    return [slugify(v) for v in variants if v]


def best_fuzzy_gated(key, tail, pool, min_ratio):
    """Fuzzy match restricted to same creator + an exact numeric-token
    match against the CANDIDATE'S NAME (not slug -- AA slugs sometimes
    drop the version number the name keeps, e.g. slug "qwen-turbo" for
    "Qwen2.5 Turbo"). Numbers must match as an ORDERED tuple, not a set:
    a naive set comparison lets "gpt-5-pro" (nums={5}) match "GPT-5.5 Pro"
    (nums={5,5}->set{5}) since sets collapse duplicates -- that silently
    swapped Sonnet 4.5 for Sonnet 5 and gpt-5 for gpt-5.5 during dev."""
    key_nums = tuple(NUM_RE.findall(slugify(tail)))
    best, best_r = None, 0.0
    for m in pool:
        cand_nums = tuple(NUM_RE.findall(slugify(name_base(m["name"]))))
        if key_nums != cand_nums:
            continue
        for cand_raw in (m["slug"], name_base(m["name"])):
            r = difflib.SequenceMatcher(None, key, alnum(cand_raw)).ratio()
            if r > best_r:
                best_r, best = r, m
    return (best, best_r) if best_r >= min_ratio else (None, best_r)


def match_catalog(catalog, aa_models):
    aa_by_slug = {m["slug"].lower(): m for m in aa_models}
    aa_by_namekey = defaultdict(list)
    aa_by_creator = defaultdict(list)
    for m in aa_models:
        aa_by_namekey[alnum(name_base(m["name"]))].append(m)
        aa_by_creator[m["model_creator"]["name"]].append(m)

    results = {}
    for row in catalog:
        suffix, tail, creator = normalize_id(row["id"])
        if suffix in results:
            continue  # already resolved via another alias of the same suffix
        key = alnum(tail)
        match, ratio, mtype = None, None, "none"

        for sv in candidate_slug_variants(tail):
            if sv in aa_by_slug:
                match, ratio, mtype = aa_by_slug[sv], 1.0, "exact_slug"
                break

        if not match:
            pool = aa_by_namekey.get(key)
            if pool:
                # ambiguous (e.g. base name has both Reasoning/Non-reasoning
                # variants): report the highest-scoring one and say so, so
                # the caller knows this may not match the provider's actual
                # default routing.
                pool_sorted = sorted(
                    pool, key=lambda m: (m["evaluations"]["artificial_analysis_intelligence_index"] is None,
                                          -(m["evaluations"]["artificial_analysis_intelligence_index"] or 0)))
                match = pool_sorted[0]
                mtype = "exact_name" if len(pool) == 1 else f"exact_name_best_of_{len(pool)}"
                ratio = 1.0

        if not match and creator:
            pool = aa_by_creator.get(creator, [])
            m, r = best_fuzzy_gated(key, tail, pool, 0.90)
            if m:
                match, ratio, mtype = m, r, "fuzzy_hi"
            else:
                m, r = best_fuzzy_gated(key, tail, pool, 0.72)
                if m:
                    match, ratio, mtype = m, r, "fuzzy_lo"  # review before trusting

        results[suffix] = {
            "tail": tail, "creator": creator, "match_type": mtype,
            "aa_name": match["name"] if match else None,
            "aa_slug": match["slug"] if match else None,
            "aa_idx": match["evaluations"]["artificial_analysis_intelligence_index"] if match else None,
            "ratio": round(ratio, 3) if ratio else None,
        }
    return results


def main():
    catalog_path = sys.argv[1] if len(sys.argv) > 1 else "catalog.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "aa_matches.json"

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    print("fetching Artificial Analysis catalog...", file=sys.stderr)
    aa_models = fetch_aa_catalog()
    print(f"AA catalog: {len(aa_models)} models", file=sys.stderr)

    results = match_catalog(catalog, aa_models)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    by_type = defaultdict(int)
    for r in results.values():
        by_type[r["match_type"]] += 1
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v}", file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    print(
        "\nNOTE: fuzzy_lo matches are string-similarity guesses and include\n"
        "known-bad patterns (e.g. same numeric tokens by coincidence, or a\n"
        "different product tier like Flash vs Plus) -- review before trusting\n"
        "them for anything more than a rough ranking. See README.md.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
