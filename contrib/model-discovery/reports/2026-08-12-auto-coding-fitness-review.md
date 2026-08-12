# auto/best-coding fitness review — 2026-08-12

Follow-up to the 2026-08-10 rating pipeline: investigated how `auto/best-coding`
(an alias for the `coding` Auto-Combo variant, `AUTO_TEMPLATE_VARIANTS` in
`open-sse/services/autoCombo/builtinCatalog.ts`) actually scores candidates, and
compared it against this pipeline's own AA/Arena-calibrated ranking.

## How auto/best-coding resolves a model's fitness

`getTaskFitnessWithSource()` (`open-sse/services/autoCombo/taskFitness.ts`) tries,
in order: `user_override` → `arena_elo` (DB `model_intelligence`, populated by the
`ARENA_ELO_SYNC_ENABLED` periodic sync, default-on) → `models_dev_tier` (derived
from `model_capabilities`, empty on this instance) → static `FITNESS_TABLE` →
wildcard boost.

On this instance `model_intelligence` **is** populated (154 rows / 61 distinct
model names, last synced 2026-08-11, source `arena_elo`) — so for the ~60 flagship
models it tracks, `auto/best-coding` already uses live, version-precise Arena Elo
data, not the static table. Spot-checking exact-name overlaps against this
pipeline's own `calibration_pairs.json` (Arena data captured from the user's saved
leaderboard export) showed agreement within 0-7 Elo points — both draw from the
same live LMArena source.

## The gap: everything outside that ~60-model set

The static `FITNESS_TABLE` (fallback layer) had not been updated for the current
model generation: no `gpt-5.5`/`gpt-5.6`, no `claude-opus-5`/`claude-sonnet-5`, no
`gemini-3.x`, no `qwen3.7`/`qwen3.8`, no `deepseek-v4`, no `kimi-k3`, no
`glm-5.2`/`minimax-m3` patterns. Most of the OmniRoute catalog — including nearly
everything this pipeline classified into `contrib/model-discovery`'s opus/sonnet/
haiku tiers — falls outside the live-synced ~60 and lands on this stale table.

Also found: the table's own un-versioned `coding` row ranked `claude-sonnet` (0.95)
above `claude-opus` (0.92), while the instance's own live `arena_elo` data ranks
Opus above Sonnet within comparable generations for coding specifically (e.g.
`claude-opus-4` 1545 vs `claude-sonnet-4-6` 1524 Elo). Left the un-versioned rows
as-is (that ranking may reflect a real, deliberate call for older generations —
Sonnet 3.5/4 were genuinely marketed as coding-strong) and instead added
version-specific rows that correctly outrank them via the existing
longest-pattern-first resolution (`getStaticFitnessTableScore`, #8603).

## Fix

Added 19 current-generation entries to `FITNESS_TABLE.coding`
(`open-sse/services/autoCombo/taskFitness.ts`), values taken directly from this
instance's own `model_intelligence` `coding`-category `score` column (the
"-high" representative variant per family, not the maxed-out `-max`/`xhigh`
submode) — same scale as the existing table (ceiling 0.98), so the fallback now
approximates what the live layer would say instead of guessing. Regression test:
`tests/unit/task-fitness-current-gen-models.test.ts`.

This is a point-in-time snapshot (2026-08-11 sync) of a fast-moving leaderboard;
re-derive from a fresh `model_intelligence` pull if it goes stale again — the
query used:

```sql
SELECT model, category, score, elo_raw, confidence
FROM model_intelligence
WHERE source = 'arena_elo'
ORDER BY category, score DESC;
```
