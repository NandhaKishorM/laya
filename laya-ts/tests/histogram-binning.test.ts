import { describe, expect, it } from "vitest";
import {
  applyBinningMap,
  checkBinningMap,
  type BinningMap,
} from "../src/common.js";
import { Agent, type ChoiceAnswer, type ScoreAnswer, type NoulAnswer } from "../src/agent.js";

const fakeProvider = (logits: number[][] = [[2.0, 0.0]], act: number[][] = [[3.0, 0.0]]) => ({
  async runEncoder(_b: any) {
    return { lastHidden: [[1, 0], [0, 1]] };
  },
  async runHead(_h: any) {
    return {
      logits,
      act,
    };
  },
});

describe("checkBinningMap", () => {
  it("validates well-formed binning maps", () => {
    const valid: BinningMap = {
      "choice:2": { bins: 5, values: [0.1, 0.3, 0.5, 0.7, 0.9] },
      "score:3-5": { bins: 2, values: [0.2, 0.8] },
    };
    expect(checkBinningMap(valid)).toEqual(valid);
  });

  it("accepts an empty binning map", () => {
    expect(checkBinningMap({})).toEqual({});
  });

  it("rejects non-objects, arrays, null and primitives", () => {
    const bads = [null, undefined, 123, "not-a-map", true, false, [1, 2, 3]];
    for (const bad of bads) {
      expect(() => checkBinningMap(bad)).toThrowError(/binning_map must be an object of bucket -> \{bins, values\}/);
    }
  });

  it("rejects invalid bucket keys that cannot match", () => {
    const badKeys = [
      { "choice:3to5": { bins: 1, values: [0.5] } },
      { "invalid:2": { bins: 1, values: [0.5] } },
      { "choice:1": { bins: 1, values: [0.5] } },
      { "default": { bins: 1, values: [0.5] } },
    ];
    for (const bad of badKeys) {
      expect(() => checkBinningMap(bad)).toThrowError(/is not a bucket like "choice:2" or "score:3-5"/);
    }
  });

  it("rejects non-object bucket entries", () => {
    const bads = [
      { "choice:2": null },
      { "choice:2": "invalid" },
      { "choice:2": [1, 2, 3] },
      { "choice:2": 42 },
    ];
    for (const bad of bads) {
      expect(() => checkBinningMap(bad)).toThrowError(/must be an object with "bins" and "values"/);
    }
  });

  it("rejects invalid bins count", () => {
    const badBins = [
      { "choice:2": { bins: 0, values: [] } },
      { "choice:2": { bins: -1, values: [] } },
      { "choice:2": { bins: 2.5, values: [0.1, 0.2] } },
      { "choice:2": { bins: "3", values: [0.1, 0.2, 0.3] } },
      { "choice:2": { bins: true, values: [0.1] } },
      { "choice:2": { values: [0.1] } },
    ];
    for (const bad of badBins) {
      expect(() => checkBinningMap(bad)).toThrowError(/must have an integer "bins" >= 1/);
    }
  });

  it("rejects mismatched values length", () => {
    const mismatches = [
      { "choice:2": { bins: 3, values: [0.1, 0.2] } },
      { "choice:2": { bins: 2, values: [0.1, 0.2, 0.3] } },
      { "choice:2": { bins: 2, values: "not-an-array" } },
    ];
    for (const bad of mismatches) {
      expect(() => checkBinningMap(bad)).toThrowError(/must have "values" of length "bins"/);
    }
  });

  it("rejects non-numbers and out of range [0, 1] numbers in values", () => {
    const badValues = [
      { "choice:2": { bins: 2, values: [0.1, "0.5"] } },
      { "choice:2": { bins: 2, values: [0.1, true] } },
      { "choice:2": { bins: 2, values: [0.1, false] } },
      { "choice:2": { bins: 2, values: [0.1, NaN] } },
      { "choice:2": { bins: 2, values: [0.1, Infinity] } },
      { "choice:2": { bins: 2, values: [-0.01, 0.5] } },
      { "choice:2": { bins: 2, values: [0.5, 1.01] } },
    ];
    for (const bad of badValues) {
      expect(() => checkBinningMap(bad)).toThrowError(/values must be numbers in \[0, 1\]/);
    }
  });
});

describe("applyBinningMap", () => {
  const map: BinningMap = {
    "choice:2": { bins: 4, values: [0.15, 0.35, 0.65, 0.95] },
    "score:2": { bins: 2, values: [0.2, 0.8] },
  };

  it("passes through confidence when map is null/undefined or bucket is missing", () => {
    expect(applyBinningMap(0.75, "choice:2", null)).toBe(0.75);
    expect(applyBinningMap(0.75, "choice:2", undefined)).toBe(0.75);
    expect(applyBinningMap(0.75, "choice:3-5", map)).toBe(0.75);
    expect(applyBinningMap(0.75, "unknown:bucket", map)).toBe(0.75);
  });

  it("passes through non-finite confidence", () => {
    expect(applyBinningMap(NaN, "choice:2", map)).toBeNaN();
    expect(applyBinningMap(Infinity, "choice:2", map)).toBe(Infinity);
  });

  it("correctly indexes bins for valid inputs", () => {
    // bins = 4: bin 0: [0, 0.25), bin 1: [0.25, 0.50), bin 2: [0.50, 0.75), bin 3: [0.75, 1.0]
    expect(applyBinningMap(0.0, "choice:2", map)).toBe(0.15);
    expect(applyBinningMap(0.1, "choice:2", map)).toBe(0.15);
    expect(applyBinningMap(0.24, "choice:2", map)).toBe(0.15);
    expect(applyBinningMap(0.25, "choice:2", map)).toBe(0.35);
    expect(applyBinningMap(0.49, "choice:2", map)).toBe(0.35);
    expect(applyBinningMap(0.50, "choice:2", map)).toBe(0.65);
    expect(applyBinningMap(0.74, "choice:2", map)).toBe(0.65);
    expect(applyBinningMap(0.75, "choice:2", map)).toBe(0.95);
    expect(applyBinningMap(0.99, "choice:2", map)).toBe(0.95);
    expect(applyBinningMap(1.0, "choice:2", map)).toBe(0.95);
  });

  it("clamps values outside [0, 1] to first and last bins", () => {
    expect(applyBinningMap(-0.5, "choice:2", map)).toBe(0.15);
    expect(applyBinningMap(1.5, "choice:2", map)).toBe(0.95);
  });
});

describe("Agent binning_map integration", () => {
  const binningMap: BinningMap = {
    "choice:2": { bins: 4, values: [0.1, 0.4, 0.7, 0.99] },
    "score:2": { bins: 2, values: [0.05, 0.95] },
    "noul:2": { bins: 2, values: [0.12, 0.88] },
  };

  it("accepts binning_map via cfg or options", () => {
    const a1 = new Agent({ provider: fakeProvider() as any, cfg: { binning_map: binningMap } });
    expect(a1.binningMap).toEqual(binningMap);

    const a2 = new Agent({ provider: fakeProvider() as any, binning_map: binningMap });
    expect(a2.binningMap).toEqual(binningMap);

    const a3 = new Agent({ provider: fakeProvider() as any, binningMap });
    expect(a3.binningMap).toEqual(binningMap);

    const a4 = new Agent({ provider: fakeProvider() as any });
    expect(a4.binningMap).toBeNull();
  });

  it("rejects invalid binning_map during Agent construction", () => {
    expect(
      () => new Agent({ provider: fakeProvider() as any, binning_map: "not-a-map" as any }),
    ).toThrowError(/binning_map must be an object/);
  });

  it("recalibrates answer_confidence on choice question", async () => {
    // logits: [2.0, 0.0] -> softmax([2, 0]) ~ [0.8808, 0.1192] -> ans_raw ~ 0.8808
    // With 4 bins: floor(0.8808 * 4) = floor(3.5232) = 3 -> values[3] = 0.99
    const agentWithBinning = new Agent({
      provider: fakeProvider([[2.0, 0.0]]) as any,
      binning_map: binningMap,
    });
    const agentWithout = new Agent({
      provider: fakeProvider([[2.0, 0.0]]) as any,
    });

    const resWith = await agentWithBinning.predict("dummy state", {
      q1: { type: "choice", instructions: "Pick one", criteria: ["A", "B"] },
    });
    const ansWith = resWith.answers.q1 as ChoiceAnswer;
    expect(ansWith.answer_confidence).toBe(0.99);

    const resWithout = await agentWithout.predict("dummy state", {
      q1: { type: "choice", instructions: "Pick one", criteria: ["A", "B"] },
    });
    const ansWithout = resWithout.answers.q1 as ChoiceAnswer;
    expect(ansWithout.answer_confidence).toBe(0.8808);
    // Uncalibrated entropy confidence remains untouched
    expect(ansWith.confidence).toBe(ansWithout.confidence);
  });

  it("recalibrates answer_confidence on score question", async () => {
    const agentWithBinning = new Agent({
      provider: fakeProvider([[2.0, 0.0]]) as any,
      binning_map: binningMap,
    });
    const res = await agentWithBinning.predict("dummy state", {
      q1: { type: "score", instructions: "Score this", criteria: ["Low", "High"] },
    });
    const ans = res.answers.q1 as ScoreAnswer;
    // max(p) ~ 0.8808 in bucket "score:2", bins=2 -> floor(0.8808 * 2) = 1 -> values[1] = 0.95
    expect(ans.answer_confidence).toBe(0.95);
  });

  it("recalibrates answer_confidence on noul question", async () => {
    const agentWithBinning = new Agent({
      provider: fakeProvider([[2.0, 0.0]]) as any,
      binning_map: binningMap,
    });
    const res = await agentWithBinning.predict("dummy state", {
      q1: { type: "noul", instructions: "Is this valid?" },
    });
    const ans = res.answers.q1 as NoulAnswer;
    // bucket "noul:2", bins=2 -> floor(0.8808 * 2) = 1 -> values[1] = 0.88
    expect(ans.answer_confidence).toBe(0.88);
  });

  it("bypasses binning recalibration when language override is active (Python parity)", async () => {
    const agent = new Agent({
      provider: fakeProvider([[2.0, 0.0]]) as any,
      binning_map: binningMap,
      lang_temperatures: {
        fr: { temperature: [1.0, 1.0, 1.0] },
      },
    });

    // Without lang override (or non-matching lang), binning recalibration applies
    const resEn = await agent.predict(
      "dummy state",
      { q1: { type: "choice", instructions: "Pick one", criteria: ["A", "B"] } },
      { lang: "en" },
    );
    expect((resEn.answers.q1 as ChoiceAnswer).answer_confidence).toBe(0.99);

    // With matching lang override "fr", binning recalibration is bypassed
    const resFr = await agent.predict(
      "dummy state",
      { q1: { type: "choice", instructions: "Pick one", criteria: ["A", "B"] } },
      { lang: "fr" },
    );
    expect((resFr.answers.q1 as ChoiceAnswer).answer_confidence).toBe(0.8808);
  });

  it("gates low confidence using recalibrated answer_confidence", async () => {
    // Without binning: ans_raw = 0.8808 >= 0.95 threshold -> low_confidence would be false.
    // With binning mapping into 0.70: 0.70 < 0.95 threshold -> low_confidence is flagged true!
    const mappingDown: BinningMap = {
      "choice:2": { bins: 1, values: [0.70] },
    };
    const agent = new Agent({
      provider: fakeProvider([[2.0, 0.0]]) as any,
      binning_map: mappingDown,
    });

    const res = await agent.predict(
      "dummy state",
      { q1: { type: "choice", instructions: "Pick one", criteria: ["A", "B"] } },
      { minConfidence: 0.85 },
    );
    const ans = res.answers.q1 as ChoiceAnswer;
    expect(ans.answer_confidence).toBe(0.70);
    expect(ans.low_confidence).toBe(true);
  });
});
