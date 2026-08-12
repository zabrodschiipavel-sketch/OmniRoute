import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { getStaticFitnessTableScore } from "../../open-sse/services/autoCombo/taskFitness.ts";

describe("taskFitness - current-generation coding fitness entries", () => {
  it("resolves current-generation Claude Opus/Sonnet ids to their own coding score", () => {
    assert.equal(getStaticFitnessTableScore("claude-opus-5-max", "coding"), 0.92);
    assert.equal(getStaticFitnessTableScore("claude-sonnet-5-high", "coding"), 0.64);
  });

  it("prefers the longer, version-specific pattern over an older un-versioned row (#8603 ordering)", () => {
    // "kimi-k2.7" (9 chars) must shadow the older un-versioned "kimi-k2" (7 chars) row.
    assert.equal(getStaticFitnessTableScore("kimi-k2.7-code", "coding"), 0.49);
    assert.notEqual(getStaticFitnessTableScore("kimi-k2.7-code", "coding"), 0.82);

    // "glm-5.2" (7 chars) must shadow the shorter "glm-5" (5 chars) row.
    assert.equal(getStaticFitnessTableScore("glm-5.2-max", "coding"), 0.74);
    assert.notEqual(getStaticFitnessTableScore("glm-5.2-max", "coding"), 0.78);
  });

  it("still falls back to the older un-versioned row for a variant with no dedicated entry", () => {
    // No "claude-opus-6" row was added, so it should still resolve via the
    // generic "claude-opus" pattern rather than returning null.
    assert.equal(getStaticFitnessTableScore("claude-opus-6", "coding"), 0.92);
  });

  it("resolves the remaining current-generation additions", () => {
    assert.equal(getStaticFitnessTableScore("claude-fable-5", "coding"), 0.84);
    assert.equal(getStaticFitnessTableScore("gpt-5.6-sol-xhigh", "coding"), 0.59);
    assert.equal(getStaticFitnessTableScore("gpt-5.5-high", "coding"), 0.51);
    assert.equal(getStaticFitnessTableScore("gemini-3.6-flash", "coding"), 0.63);
    assert.equal(getStaticFitnessTableScore("gemini-3.5-flash-high", "coding"), 0.56);
    assert.equal(getStaticFitnessTableScore("gemini-3.1-pro-preview", "coding"), 0.42);
    assert.equal(getStaticFitnessTableScore("gemini-3-pro", "coding"), 0.4);
    assert.equal(getStaticFitnessTableScore("qwen3.8-max", "coding"), 0.93);
    assert.equal(getStaticFitnessTableScore("qwen3.7-max-20260517", "coding"), 0.58);
    assert.equal(getStaticFitnessTableScore("qwen3.6-max-preview", "coding"), 0.5);
    assert.equal(getStaticFitnessTableScore("deepseek-v4-flash-high", "coding"), 0.74);
    assert.equal(getStaticFitnessTableScore("deepseek-v4-pro-high-preview", "coding"), 0.46);
    assert.equal(getStaticFitnessTableScore("kimi-k3-max", "coding"), 0.94);
    assert.equal(getStaticFitnessTableScore("kimi-k2.6", "coding"), 0.57);
    assert.equal(getStaticFitnessTableScore("minimax-m3", "coding"), 0.52);
    assert.equal(getStaticFitnessTableScore("mimo-v2.5-pro", "coding"), 0.49);
  });
});
