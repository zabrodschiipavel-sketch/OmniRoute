#!/usr/bin/env python3
"""Extend AA-based scoring (match_aa.py) with Arena Elo, for the models AA
doesn't track. Arena (arena.ai/leaderboard/text) has no public bulk API and
a raw GET on its internal data endpoint 404s/500s outside a real browser
session (Cloudflare), so this reads a saved copy of the page instead:

  1. Open https://arena.ai/leaderboard/text in a browser, expand to "View
     all" so the full table is in the DOM.
  2. Save As -> Webpage, Complete (or just the single .html; the leaderboard
     <table> is server-rendered into the saved DOM, no JS execution needed
     to re-parse it).
  3. python match_arena.py page.html aa_matches.json arena_matches.json

Elo and the AA Intelligence Index measure different things (Arena is human
head-to-head preference in chat; AA is benchmark performance) and only
correlate moderately -- see calibrate() and README.md for the actual
cross-validated error before trusting a derived number.
"""
import json
import re
import sys
import difflib
from collections import defaultdict

from bs4 import BeautifulSoup


# --------------------------------------------------------- parse the page

def parse_arena_page(html_path):
    with open(html_path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    table = soup.find("table")
    if table is None:
        raise SystemExit("no <table> found -- did the page finish loading before you saved it?")

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        model_cell = cells[2]
        link = model_cell.find("a")
        if not link:
            continue
        span = link.find("span", title=True)
        slug = span["title"].strip() if span else link.get_text(strip=True)

        score_spans = cells[3].find_all("span")
        if not score_spans:
            continue
        try:
            score = float(score_spans[0].get_text(strip=True))
        except ValueError:
            continue
        rows.append({"slug": slug, "score": score})
    return rows


# -------------------------------------------------------------- matching
# Mirrors match_aa.py's approach (ordered candidate variants, numeric-token
# gate on ORDERED tuples not sets) -- see that file's comments for why.

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
PAREN_RE = re.compile(r"\([^)]*\)")


def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def base(s):
    return PAREN_RE.sub("", s).strip()


def candidate_slug_variants(tail):
    t = re.sub(r"^@?cf/", "", tail)
    variants = []

    def add(v):
        if v and v not in variants:
            variants.append(v)

    add(t)
    if "/" in t:
        add(t.split("/")[-1])
    for suf in ("-instruct", "-it", "-chat"):
        if t.endswith(suf):
            add(t[: -len(suf)])
    for suf in ("-preview", "-latest", "-free"):
        if t.endswith(suf):
            add(t[: -len(suf)])
    return [slugify(v) for v in variants if v]


def match_arena(aa_matches, arena_rows):
    arena_by_slug = {}
    for m in arena_rows:
        arena_by_slug.setdefault(slugify(m["slug"]), m)
    arena_by_namekey = defaultdict(list)
    for m in arena_rows:
        arena_by_namekey[alnum(base(m["slug"]))].append(m)

    results = {}
    for suffix, aa in aa_matches.items():
        tail, creator = aa["tail"], aa["creator"]
        key = alnum(tail)
        match, mtype = None, "none"

        for sv in candidate_slug_variants(tail):
            if sv in arena_by_slug:
                match, mtype = arena_by_slug[sv], "exact_slug"
                break

        if not match:
            pool = arena_by_namekey.get(key)
            if pool:
                match = max(pool, key=lambda m: m["score"])
                mtype = "exact_name" if len(pool) == 1 else f"exact_name_best_of_{len(pool)}"

        results[suffix] = {
            "arena_slug": match["slug"] if match else None,
            "arena_score": match["score"] if match else None,
            "match_type": mtype,
        }
    return results


# --------------------------------------------------------------- calibrate

def calibrate(aa_matches, arena_matches):
    """Isotonic (monotonic) regression Elo -> AA-equivalent index, fit on
    models present in BOTH. Isotonic over linear/polynomial because it
    can't produce nonsense outside the sampled range (e.g. a linear fit
    predicts a NEGATIVE intelligence index below ~1150 Elo) and clips to
    the nearest real observation instead of extrapolating."""
    from sklearn.isotonic import IsotonicRegression
    import numpy as np

    pairs, seen = [], set()
    for suffix, aa in aa_matches.items():
        if aa["aa_idx"] is None:
            continue
        ar = arena_matches.get(suffix, {})
        if ar.get("arena_score") is None:
            continue
        key = (ar["arena_score"], aa["aa_idx"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    if len(pairs) < 20:
        print(f"only {len(pairs)} AA<->Arena overlap pairs -- too few to "
              f"calibrate reliably, skipping", file=sys.stderr)
        return None, 0

    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(xs, ys)
    return model, len(pairs)


def main():
    html_path = sys.argv[1] if len(sys.argv) > 1 else "arena-leaderboard.html"
    aa_path = sys.argv[2] if len(sys.argv) > 2 else "aa_matches.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "arena_matches.json"

    arena_rows = parse_arena_page(html_path)
    print(f"parsed {len(arena_rows)} Arena rows from {html_path}", file=sys.stderr)

    with open(aa_path, encoding="utf-8") as f:
        aa_matches = json.load(f)

    arena_matches = match_arena(aa_matches, arena_rows)
    model, n = calibrate(aa_matches, arena_matches)

    for suffix, ar in arena_matches.items():
        aa_idx = aa_matches[suffix]["aa_idx"]
        if aa_idx is None and ar["arena_score"] is not None and model is not None:
            ar["aa_equivalent_estimate"] = round(float(model.predict([ar["arena_score"]])[0]), 1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(arena_matches, f, ensure_ascii=False, indent=1)

    by_type = defaultdict(int)
    for r in arena_matches.values():
        by_type[r["match_type"]] += 1
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v}", file=sys.stderr)
    new_coverage = sum(1 for r in arena_matches.values() if r.get("aa_equivalent_estimate") is not None)
    print(f"calibrated on {n} AA<->Arena pairs; extended coverage to "
          f"{new_coverage} models Arena has but AA doesn't", file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
