import { describe, expect, it } from "vitest";
import { checkQuestion } from "../src/agent.js";
import { q, best, topK, isConfident } from "../src/dx.js";
import { predictShortlist } from "../src/shortlist.js";

describe("q builders", () => {
  it("produce valid defs", () => {
    const c: any = q.choice("What?", { a: "x", b: "y" });
    const s: any = q.score("Rate?", ["bad", "ok"]);
    const n: any = q.noul("Holds?");
    expect(() => checkQuestion("c", c)).not.toThrow();
    expect(() => checkQuestion("s", s)).not.toThrow();
    expect(() => checkQuestion("n", n)).not.toThrow();
    expect(c.criteria).toEqual({ a: "x", b: "y" });
  });
  it("choice accepts label list", () => {
    const c: any = q.choice("Pick?", ["a", "b"]);
    expect(() => checkQuestion("c", c)).not.toThrow();
  });
});

describe("result helpers", () => {
  it("best/topK/isConfident", () => {
    expect(best({ type: "choice", choice: "b", probabilities: { a: 0.2, b: 0.8 } } as any)).toBe("b");
    expect(best({ type: "score", probabilities: { "0": 0.1, "1": 0.7, "2": 0.2 } } as any)).toBe("1");
    expect(topK({ a: 0.2, b: 0.8, c: 0.5 }, 2)).toEqual(["b", "c"]);
    expect(isConfident({ answer_confidence: 0.9 } as any, 0.8)).toBe(true);
    expect(isConfident({ answer_confidence: 0.5 } as any, 0.8)).toBe(false);
  });
  it("gates on answer_confidence alone, ignoring entropy confidence", () => {
    expect(isConfident({ confidence: 0.5, answer_confidence: 0.9 } as any, 0.8)).toBe(true);
    expect(isConfident({ confidence: 0.9, answer_confidence: 0.5 } as any, 0.8)).toBe(false);
    expect(isConfident({ confidence: 0.9 } as any, 0.8)).toBe(false);
  });
});

describe("predictShortlist auto-embed", () => {
  it("uses agent encoder when embedFn omitted", async () => {
    const agent: any = {
      tok: { encode: (t: string) => [t.length % 100], padId: 0 },
      provider: {
        async runEncoder(batch: any) {
          return { lastHidden: batch.inputIds.map((row: number[]) => row.map(() => [1, 0])) };
        },
      },
      async predict(_s: unknown, reduced: any) {
        return { model: "m", answers: {}, usage: { input_tokens: 0, output_tokens: 0 }, _reduced: reduced };
      },
    };
    const questions: any = {
      d: { type: "choice", instructions: "pick", criteria: { aaa: "x", bbb: "y", ccc: "z" } },
    };
    const out: any = await predictShortlist(agent, "state", questions, undefined as any, 1);
    expect(out.shortlist.d.labels).toHaveLength(1);
  });
});
