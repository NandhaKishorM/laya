// A question's `instructions` is the text the model is asked, so a null or empty one is not a
// weak question, it is a question with nothing to answer. Python's `Agent._check_question`
// rejects null, a blank string, and an empty list or dict; the TypeScript port only checked that
// the key was present, and `toInternal` then serialised whatever it found, so `null` reached the
// model as the literal prompt "null" and `{}` as "{}".
import { describe, expect, it } from "vitest";
import { Agent, checkQuestion } from "../src/agent.js";

const fakeProvider = () => ({
  async runEncoder(b: any) {
    const n = b?.inputIds?.length ?? 1;
    return { lastHidden: Array.from({ length: n }, () => [[1, 0], [0, 1]]) };
  },
  async runHead(_h: any, batch?: any) {
    const n = batch?.inputIds?.length ?? 1;
    return {
      logits: Array.from({ length: n }, () => [2, 0, 1, 0]),
      act: Array.from({ length: n }, () => [3, 0]),
    };
  },
});

const CHOICE = { type: "choice", criteria: { x: "yes", y: "no" } };

// Python: agent.py _check_question, the three rejections in a row.
const REJECTED: Array<[string, unknown]> = [
  ["null", null],
  ["undefined", undefined],
  ["an empty string", ""],
  ["a blank string", "   "],
  ["an empty list", []],
  ["an empty dict", {}],
];

describe("checkQuestion instructions", () => {
  it.each(REJECTED)("rejects %s", (_label, ins) => {
    expect(() => checkQuestion("q", { ...CHOICE, instructions: ins })).toThrow(
      /'instructions' must not be (None|empty)/,
    );
  });

  it("names the question and what to fix", () => {
    expect(() => checkQuestion("q", { ...CHOICE, instructions: null })).toThrow(
      'question "q": \'instructions\' must not be None',
    );
  });

  it("accepts instructions that can be answered", () => {
    for (const ins of ["Is it urgent?", "  padded  ", 42, { k: "v" }, ["a", "b"]]) {
      expect(() => checkQuestion("q", { ...CHOICE, instructions: ins })).not.toThrow();
    }
  });

  it("still reports a missing key as missing", () => {
    expect(() => checkQuestion("q", { ...CHOICE })).toThrow("no 'instructions'");
  });
});

describe("checkQuestion id", () => {
  it("rejects a blank question id, as Python does", () => {
    for (const qid of ["", "   "]) {
      expect(() => checkQuestion(qid, { ...CHOICE, instructions: "q?" })).toThrow(
        "question id must be a non-empty string",
      );
    }
  });
});

describe("Agent", () => {
  it("rejects an unanswerable question instead of prompting with it", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    for (const method of ["predict", "systemOne"] as const) {
      for (const [_label, ins] of REJECTED) {
        await expect(
          a[method]("hello", { q: { ...CHOICE, instructions: ins } } as any),
        ).rejects.toThrow('question "q": \'instructions\' must not be');
      }
    }
  });

  it("still answers a question whose instructions are real", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const r: any = await a.systemOne("hi", {
      q: { ...CHOICE, instructions: "Is it urgent?" },
    });
    expect(r.answers.q.choice).toBe("x");
  });
});
