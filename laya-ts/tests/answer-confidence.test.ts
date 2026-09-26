import { describe, expect, it } from "vitest";
import { answerConfidence, confidenceFromProbs } from "../src/common.js";
import { Agent } from "../src/agent.js";

const fakeProvider = (n: number) => ({
  // n logits with a known argmax at 0; softmax gives a fixed, checkable distribution
  async runEncoder(_b: any) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: any) {
    const row = Array.from({ length: n }, (_, i) => (i === 0 ? 2 : 0));
    return { logits: [row], act: [[3, 0]] };
  },
});

function probs(provider: any, a: Agent) { return a as any; }

describe("answerConfidence", () => {
  it("is max(p) on the reported answer", () => {
    expect(answerConfidence([0.25, 0.6, 0.15])).toBeCloseTo(0.6);
    expect(answerConfidence([1.0])).toBe(1.0);
  });

  it("returns 1.0 for an empty distribution (Python parity: k < 1)", () => {
    expect(answerConfidence([])).toBe(1.0);
  });

  it("stays at the same value across option counts, unlike the entropy confidence", () => {
    // The #361 table: same top probability 0.9, different label-set sizes.
    const fill = (k: number) => {
      const rest = Array.from({ length: k - 1 }, () => 0.1 / (k - 1));
      return [0.9, ...rest];
    };
    for (const k of [2, 5, 20, 77]) {
      expect(answerConfidence(fill(k))).toBeCloseTo(0.9, 5);
    }
    // entropy confidence rises with k for the same max(p)
    expect(confidenceFromProbs(fill(2))).toBeLessThan(confidenceFromProbs(fill(20)));
  });

  it("is attached to every answer type by the agent decode", async () => {
    const mk = (n: number) => new Agent({ provider: fakeProvider(n) } as any);
    const choice: any = await mk(3).systemOne("s", {
      c: { type: "choice", instructions: "q?", criteria: { a: "x", b: "y", c: "z" } },
    });
    const top = Math.max(...Object.values(choice.answers.c.probabilities).map(Number));
    expect(choice.answers.c.answer_confidence).toBeCloseTo(top, 4);

    const score: any = await mk(3).systemOne("s", {
      s: { type: "score", instructions: "q?", criteria: ["lo", "mid", "hi"] },
    });
    expect(score.answers.s.answer_confidence).toBeCloseTo(top, 4);

    const noul: any = await mk(2).systemOne("s", { n: { type: "noul", instructions: "q?" } });
    // over two options max(p) equals the existing noul confidence
    expect(noul.answers.n.answer_confidence).toBeCloseTo(noul.answers.n.confidence, 4);
  });
});
