import { describe, expect, it } from "vitest";
import {
  checkMinConfidence,
  checkMinConfidenceMap,
  flagLowConfidence,
  optionBucket,
  resolveMinConfidence,
} from "../src/common.js";
import { Agent, type ChoiceAnswer } from "../src/agent.js";
import { Router } from "../src/router.js";

const fakeProvider = () => ({
  async runEncoder(_b: any) {
    return { lastHidden: [[1, 0], [0, 1]] };
  },
  async runHead(_h: any) {
    // 2-option question with clear margin: argmax 0
    return {
      logits: [[2.0, 0.0]],
      act: [[3.0, 0.0]],
    };
  },
});

describe("checkMinConfidence", () => {
  it("validates real numbers in [0.0, 1.0]", () => {
    for (const valid of [0.0, 0.5, 1.0, 0, 1, 0.85]) {
      expect(checkMinConfidence(valid)).toBe(Number(valid));
    }
  });

  it("rejects booleans, out of range numbers, and non-finite values", () => {
    const invalids = [
      true,
      false,
      -0.01,
      1.01,
      -1.0,
      2.0,
      NaN,
      Infinity,
      -Infinity,
      "0.5",
      null,
      undefined,
      [0.5],
    ];
    for (const invalid of invalids) {
      expect(() => checkMinConfidence(invalid)).toThrowError(/min_confidence must be a float in \[0\.0, 1\.0\]/);
    }
  });

  it("accepts and validates a per-bucket threshold map", () => {
    const map = { "choice:2": 0.9, default: 0.5 };
    expect(checkMinConfidence(map)).toEqual(map);
  });

  it("rejects empty min_confidence map", () => {
    expect(() => checkMinConfidence({})).toThrowError(/a min_confidence map must be a non-empty dict/);
  });
});

describe("checkMinConfidenceMap", () => {
  it("rejects non-object or non-dict inputs", () => {
    for (const invalid of [null, undefined, 42, "choice:2", [0.5], true]) {
      expect(() => checkMinConfidenceMap(invalid)).toThrowError(/a min_confidence map must be a non-empty dict/);
    }
  });

  it("rejects empty dict", () => {
    expect(() => checkMinConfidenceMap({})).toThrowError(/a min_confidence map must be a non-empty dict of bucket -> float, got \{\}/);
  });

  it("validates valid maps", () => {
    const map = { "choice:2": 0.9, "choice:11+": 0.5, "noul:2": 0.8, default: 0.3 };
    expect(checkMinConfidenceMap(map)).toEqual(map);
  });

  it("rejects invalid map values", () => {
    expect(() => checkMinConfidenceMap({ "choice:2": 1.5 })).toThrowError(/min_confidence must be a float in \[0\.0, 1\.0\]/);
    expect(() => checkMinConfidenceMap({ "choice:2": -0.1 })).toThrowError(/min_confidence must be a float in \[0\.0, 1\.0\]/);
    expect(() => checkMinConfidenceMap({ "choice:2": true })).toThrowError(/min_confidence must be a float in \[0\.0, 1\.0\]/);
    expect(() => checkMinConfidenceMap({ "choice:2": "0.9" as any })).toThrowError(/min_confidence must be a float in \[0\.0, 1\.0\]/);
    expect(() => checkMinConfidenceMap({ "choice:2": NaN })).toThrowError(/min_confidence must be a float in \[0\.0, 1\.0\]/);
  });

  it("rejects unknown bucket keys and typos", () => {
    for (const typo of ["choice:3_5", "Choice:2", "invalid", "score:1", "foo:2", "choice:12"]) {
      expect(() => checkMinConfidenceMap({ [typo]: 0.5 })).toThrowError(
        /min_confidence map keys must be strings like 'choice:3-5'/,
      );
    }
  });
});

describe("optionBucket & resolveMinConfidence", () => {
  it("computes optionBucket correctly", () => {
    expect(optionBucket({ type: "choice", probabilities: { a: 0.5, b: 0.5 } })).toBe("choice:2");
    expect(optionBucket({ type: "choice", probabilities: { a: 0.4, b: 0.3, c: 0.3 } })).toBe("choice:3-5");
    expect(optionBucket({ type: "choice", probabilities: { a: 0.2, b: 0.2, c: 0.2, d: 0.2, e: 0.2 } })).toBe("choice:3-5");
    const six: Record<string, number> = {};
    for (let i = 0; i < 6; i++) six[String(i)] = 1 / 6;
    expect(optionBucket({ type: "choice", probabilities: six })).toBe("choice:6-10");
    const ten: Record<string, number> = {};
    for (let i = 0; i < 10; i++) ten[String(i)] = 0.1;
    expect(optionBucket({ type: "choice", probabilities: ten })).toBe("choice:6-10");
    const eleven: Record<string, number> = {};
    for (let i = 0; i < 11; i++) eleven[String(i)] = 1 / 11;
    expect(optionBucket({ type: "choice", probabilities: eleven })).toBe("choice:11+");

    expect(optionBucket({ type: "score", probabilities: { "1": 0.3, "2": 0.3, "3": 0.4 } })).toBe("score:3-5");
    expect(optionBucket({ type: "noul" })).toBe("noul:2");
    expect(optionBucket({ type: "text" })).toBeNull();
    expect(optionBucket({ type: "choice" })).toBeNull();
  });

  it("resolves per-bucket thresholds with default and unconfigured fallback", () => {
    const MAP = { "choice:2": 0.9, "choice:11+": 0.5, "noul:2": 0.8, default: 0.3 };
    const ans = (type: string, k: number) => {
      const probs: Record<string, number> = {};
      for (let i = 0; i < k; i++) probs[String(i)] = 1 / k;
      return { type, probabilities: probs };
    };

    expect(resolveMinConfidence(ans("choice", 2), MAP)).toBe(0.9);
    expect(resolveMinConfidence(ans("choice", 12), MAP)).toBe(0.5);
    expect(resolveMinConfidence(ans("choice", 4), MAP)).toBe(0.3); // falls to default
    expect(resolveMinConfidence({ type: "noul" }, MAP)).toBe(0.8);
    // unconfigured bucket and no default in map -> returns defaultVal (0.0)
    expect(resolveMinConfidence(ans("score", 7), { "choice:2": 0.9 })).toBe(0.0);
  });
});

describe("flagLowConfidence", () => {
  it("is a no-op when minConfidence is 0.0", () => {
    const sample = [
      {
        answers: {
          q1: { type: "choice", choice: "a", answer_confidence: 0.1, confidence: 0.1 },
        },
      },
    ];
    flagLowConfidence(sample as any, 0.0);
    expect(sample[0].answers.q1).not.toHaveProperty("low_confidence");
  });

  it("flags answers where answer_confidence falls below threshold", () => {
    const sample = [
      {
        answers: {
          q_high: { type: "choice", choice: "a", answer_confidence: 0.92, confidence: 0.8 },
          q_low: { type: "score", score: 1, answer_confidence: 0.45, confidence: 0.4 },
          q_fallback: { type: "choice", choice: "b", confidence: 0.3 },
          q_exact: { type: "choice", choice: "c", answer_confidence: 0.7 },
        },
      },
    ];

    flagLowConfidence(sample as any, 0.7);
    const ans = sample[0].answers as any;

    expect(ans.q_low.low_confidence).toBe(true);
    expect(ans.q_fallback.low_confidence).toBe(true); // falls back to confidence 0.3 < 0.7
    expect(ans.q_high.low_confidence).toBeUndefined();
    expect(ans.q_exact.low_confidence).toBeUndefined(); // 0.70 >= 0.70

    // Raw answers and confidence fields remain intact
    expect(ans.q_low.answer_confidence).toBe(0.45);
    expect(ans.q_high.choice).toBe("a");
  });

  it("handles a single result object as well as an array", () => {
    const single = {
      answers: {
        q: { type: "choice", choice: "x", answer_confidence: 0.4 },
      },
    };
    flagLowConfidence(single as any, 0.5);
    expect((single.answers.q as any).low_confidence).toBe(true);
  });

  it("gates with a per-bucket map", () => {
    const MAP = { "choice:2": 0.9, "choice:11+": 0.5, default: 0.3 };
    const sample = [
      {
        answers: {
          a: { type: "choice", probabilities: { "0": 0.5, "1": 0.5 }, answer_confidence: 0.85 },
          b: {
            type: "choice",
            probabilities: Object.fromEntries(Array.from({ length: 12 }, (_, i) => [String(i), 1 / 12])),
            answer_confidence: 0.6,
          },
          c: {
            type: "choice",
            probabilities: { "0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25 },
            answer_confidence: 0.35,
          },
          d: {
            type: "score",
            probabilities: Object.fromEntries(Array.from({ length: 7 }, (_, i) => [String(i), 1 / 7])),
            answer_confidence: 0.05,
          },
        },
      },
    ];

    flagLowConfidence(sample as any, MAP);
    const ans = sample[0].answers as any;

    expect(ans.a.low_confidence).toBe(true); // 0.85 < 0.9 (choice:2)
    expect(ans.b.low_confidence).toBeUndefined(); // 0.60 >= 0.5 (choice:11+)
    expect(ans.c.low_confidence).toBeUndefined(); // 0.35 >= 0.3 (default)
    expect(ans.d.low_confidence).toBe(true); // 0.05 < 0.3 (falls back to default)

    // Unconfigured bucket with no default in map falls back to 0.0 (gate nothing)
    const unmapped = [
      {
        answers: {
          x: {
            type: "score",
            probabilities: Object.fromEntries(Array.from({ length: 7 }, (_, i) => [String(i), 1 / 7])),
            answer_confidence: 0.05,
          },
        },
      },
    ];
    flagLowConfidence(unmapped as any, { "choice:2": 0.9 });
    expect((unmapped[0].answers.x as any).low_confidence).toBeUndefined();
  });
});

describe("Agent confidence gating", () => {
  it("flags low-confidence answers when minConfidence is set", async () => {
    const agent = new Agent({ provider: fakeProvider() } as any);
    const questions = {
      choice_q: {
        type: "choice" as const,
        instructions: "Pick one",
        criteria: { a: "option A", b: "option B" },
      },
    };

    // Threshold 0.999 is above the predicted probability (softmax([2, 0]) ~ 0.88)
    const result = await agent.systemOne("hello", questions, { minConfidence: 0.99 });
    expect(result.answers.choice_q.low_confidence).toBe(true);
    expect((result.answers.choice_q as ChoiceAnswer).choice).toBe("a");

    // Threshold 0.5 is below the predicted probability
    const resultHigh = await agent.systemOne("hello", questions, { minConfidence: 0.5 });
    expect(resultHigh.answers.choice_q.low_confidence).toBeUndefined();
  });

  it("supports python parity alias min_confidence", async () => {
    const agent = new Agent({ provider: fakeProvider() } as any);
    const questions = {
      choice_q: {
        type: "choice" as const,
        instructions: "Pick one",
        criteria: { a: "option A", b: "option B" },
      },
    };

    const result = await agent.predict("hello", questions, { min_confidence: 0.99 });
    expect(result.answers.choice_q.low_confidence).toBe(true);
  });

  it("supports per-bucket threshold maps in Agent.predict", async () => {
    const agent = new Agent({ provider: fakeProvider() } as any);
    const questions = {
      choice_q: {
        type: "choice" as const,
        instructions: "Pick one",
        criteria: { a: "option A", b: "option B" },
      },
    };

    // 2-option question matches choice:2 bucket
    const resultFlagged = await agent.predict("hello", questions, { minConfidence: { "choice:2": 0.99 } });
    expect(resultFlagged.answers.choice_q.low_confidence).toBe(true);

    const resultPassed = await agent.predict("hello", questions, { minConfidence: { "choice:2": 0.5 } });
    expect(resultPassed.answers.choice_q.low_confidence).toBeUndefined();

    const resultDefault = await agent.predict("hello", questions, { min_confidence: { default: 0.99 } });
    expect(resultDefault.answers.choice_q.low_confidence).toBe(true);
  });

  it("validates minConfidence before running inference", async () => {
    const agent = new Agent({ provider: fakeProvider() } as any);
    await expect(
      agent.predict("hello", {}, { minConfidence: 1.5 }),
    ).rejects.toThrowError(/min_confidence must be a float/);
  });
});

describe("Router confidence gating", () => {
  it("flags low-confidence answers in Router.predict and supports alias", async () => {
    const router = new Router();
    // Stub router.load to return an agent using our fakeProvider
    router.load = async () => new Agent({ provider: fakeProvider() } as any);

    const questions = {
      q: {
        type: "choice" as const,
        instructions: "Pick",
        criteria: { a: "A", b: "B" },
      },
    };

    const res = await router.predict("state text", questions, { minConfidence: 0.99 });
    expect(res.answers.q.low_confidence).toBe(true);
    expect(res.routing).toBeDefined();

    const resAlias = await router.predict("state text", questions, { min_confidence: 0.99 });
    expect(resAlias.answers.q.low_confidence).toBe(true);

    const resPass = await router.predict("state text", questions, { minConfidence: 0.5 });
    expect(resPass.answers.q.low_confidence).toBeUndefined();
  });

  it("supports per-bucket threshold maps in Router.predict", async () => {
    const router = new Router();
    router.load = async () => new Agent({ provider: fakeProvider() } as any);

    const questions = {
      q: {
        type: "choice" as const,
        instructions: "Pick",
        criteria: { a: "A", b: "B" },
      },
    };

    const resFlagged = await router.predict("state text", questions, { minConfidence: { "choice:2": 0.99 } });
    expect(resFlagged.answers.q.low_confidence).toBe(true);

    const resPassed = await router.predict("state text", questions, { min_confidence: { "choice:2": 0.5 } });
    expect(resPassed.answers.q.low_confidence).toBeUndefined();
  });

  it("validates minConfidence in Router.predict before routing or inference", async () => {
    const router = new Router();
    await expect(
      router.predict("state text", {}, { minConfidence: -0.5 }),
    ).rejects.toThrowError(/min_confidence must be a float/);
  });
});
