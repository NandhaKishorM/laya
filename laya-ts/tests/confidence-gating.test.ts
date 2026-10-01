import { describe, expect, it } from "vitest";
import { checkMinConfidence, flagLowConfidence } from "../src/common.js";
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
      {},
    ];
    for (const invalid of invalids) {
      expect(() => checkMinConfidence(invalid)).toThrowError(/min_confidence must be a float in \[0\.0, 1\.0\]/);
    }
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

  it("validates minConfidence in Router.predict before routing or inference", async () => {
    const router = new Router();
    await expect(
      router.predict("state text", {}, { minConfidence: -0.5 }),
    ).rejects.toThrowError(/min_confidence must be a float/);
  });
});
