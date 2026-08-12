# Model discovery: rating- and tool-use-based model selection

OmniRoute's `/v1/models` tells you a model *exists* and what capability
flags it *claims*. It doesn't tell you whether the model is any good,
whether it can actually do function calling, or whether the provider
behind it currently has quota. This is a small pipeline that answers all
three, so you can point an agent (or a `modelRoles`/fallback config in any
agent CLI) at "the best model I can actually use right now, with real tool
use" instead of guessing from a name.

Five independent scripts, each reading the previous one's JSON output:

```
fetch_catalog.py   -> catalog.json          every model your gateway exposes
match_aa.py         -> aa_matches.json       + Artificial Analysis Intelligence Index
match_arena.py       -> arena_matches.json    + Arena Elo (calibrated to the AA scale)
smoke_test.py        -> smoke_results.json    + does it actually answer right now
tool_use_test.py     -> toolcall_results.json + does it actually call a function
```

## Setup

```bash
pip install requests beautifulsoup4 lxml scikit-learn numpy
```

Environment variables:

| Var | Used by | What |
|---|---|---|
| `OMNIROUTE_BASE_URL` | fetch_catalog, smoke_test, tool_use_test | gateway URL, default `http://127.0.0.1:20128` |
| `OMNIROUTE_API_KEY` | fetch_catalog, smoke_test, tool_use_test | a local gateway key (Dashboard -> API Keys) |
| `AA_API_KEY` | match_aa | [artificialanalysis.ai](https://artificialanalysis.ai) Data API key (free tier: 100 req/day) |

## Running it

```bash
python fetch_catalog.py catalog.json
python match_aa.py catalog.json aa_matches.json

# Arena has no scrapable bulk API (Cloudflare-protected internal endpoint) --
# save the leaderboard page yourself: open https://arena.ai/leaderboard/text,
# click "View all" so every row is in the DOM, then Save As a complete
# webpage. Point match_arena.py at the saved .html.
python match_arena.py arena-leaderboard.html aa_matches.json arena_matches.json

python smoke_test.py catalog.json smoke_results.json

# feed it only the ids that actually answered -- no point tool-testing a
# model that's quota-exhausted right now
python -c "
import json
r = json.load(open('smoke_results.json', encoding='utf-8'))
open('ok_ids.txt', 'w', encoding='utf-8').write('\n'.join(x['id'] for x in r if x['kind'] in ('ok','ok_reasoning_only')))
"
python tool_use_test.py ok_ids.txt toolcall_results.json
```

Join the four JSON files on model id / suffix to build a final ranked
table: intelligence score (AA measured or Arena-derived), smoke status,
tool-use status. `reports/` has a worked example.

## Why the matching is its own file (and why it's careful)

Matching a gateway id like `openrouter/deepseek/deepseek-v4-flash-0731` to
an AA/Arena catalog entry sounds like a string-similarity problem, and a
naive `difflib.SequenceMatcher` threshold *looks* like it works right up
until it confidently swaps `claude-sonnet-4.5` for `claude-sonnet-5`, or
`gpt-5` for `gpt-5.5`, because both pairs score >0.85 on pure character
overlap. `match_aa.py`/`match_arena.py` gate every fuzzy candidate on an
**ordered, duplicate-sensitive tuple of the numbers** in the candidate's
full name (not slug — see the code comment on why slug isn't safe either).
That single check eliminated the majority of wrong matches found during
development. It still isn't perfect: matches below `ratio < 0.90` are
tagged `fuzzy_lo` in the output and are goodenough for a rough ranking,
good enough for a rough ranking, not for anything where getting the exact
model wrong would matter.

Other things worth knowing before trusting the numbers:

- **AA vs Arena are different measurements.** AA's Intelligence Index is a
  benchmark composite (math, code, general knowledge); Arena Elo is human
  head-to-head preference in open-ended chat. They correlate (`match_arena.py`
  calibrates one to the other with isotonic regression) but not tightly —
  cross-validated error is roughly ±7 points on AA's 0–65 scale. Treat a
  Arena-derived estimate as "same tier", not "this exact number".
- **When a gateway id doesn't specify a reasoning/effort tier**, both
  matchers report the *highest-scoring* variant of that base model, which
  may not be what the provider actually serves by default if their default
  is a cheaper tier.
- **A model can claim `tool_calling: true` and still not call tools** through
  a given gateway/backend combination (observed on this catalog: a raw
  open-weight model on one hosting backend described *wanting* to call the
  function in prose instead of emitting a real `tool_calls` entry). That's
  exactly why `tool_use_test.py` sends a real request instead of trusting
  the flag.
- **`smoke_test.py`'s `quota_short`/`quota_long`/`auth_or_balance` are not
  "broken"** — they mean the provider behind that id currently has no
  quota/balance, which can resolve on its own in seconds, hours, or days
  depending on the provider. Only `bad_request`/`not_found`/`http_error`
  point at an actual misconfiguration.
