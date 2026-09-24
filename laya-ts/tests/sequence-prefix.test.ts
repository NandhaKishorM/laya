// laya-ts/tests/sequence-prefix.test.ts — buildQuestionPrefix/sequenceWithState must compose
// into exactly what buildSequence produced as one pass, and Agent must encode the state once.
import { describe, expect, it } from "vitest";
import { Agent, defaultTokenizer } from "../src/agent.js";
import {
  buildQuestionPrefix,
  buildSequence,
  renderOptions,
  sequenceWithState,
  serializeState,
  type InternalQ,
} from "../src/common.js";

const QUESTIONS: Array<[string, InternalQ]> = [
  ["choice", { t: "choice", ins: "What does the customer want?", crit: { refund: "money back", apology: "", other: "rest" } }],
  ["score", { t: "score", ins: "Rate satisfaction", crit: ["terrible", "ok", "great"] }],
  ["noul-empty", { t: "noul", ins: "The statement holds", crit: {} }],
  ["noul-custom", { t: "noul", ins: "Angry?", crit: { true: "yes, furious", false: "no" } }],
];

const LONG = "charged twice on my card and nobody replied to my tickets ".repeat(40);

const STATES: Array<[string, unknown]> = [
  ["short", "refund please"],
  ["long", LONG],
  ["object", { body: "charged twice", meta: { order: 88121, tags: ["billing", "urgent"] } }],
  ["mask-token", "why is [MASK] in my email [MASK] body?"],
];

const GEOMETRIES: Array<[string, number, number, number[] | undefined, boolean]> = [
  ["default", 512, 192, undefined, false],
  ["state-truncated", 64, 192, undefined, false],
  ["truncate-left", 64, 192, undefined, true],
  ["tight-head", 128, 40, undefined, false],
  ["prefix-overflow", 48, 192, undefined, false],
  ["reordered", 512, 192, [2, 0, 1], false],
];

describe("question prefix composition", () => {
  for (const [qn, q] of QUESTIONS) {
    for (const [sn, state] of STATES) {
      for (const [gn, maxLen, headMaxLen, order, left] of GEOMETRIES) {
        if (order && q.t !== "choice") continue; // optionOrder only applies to choice here
        it(`${qn} / ${sn} / ${gn} matches one-pass buildSequence`, () => {
          const tok = defaultTokenizer();
          const expected = buildSequence(tok, state, q, maxLen, headMaxLen, order, left);
          const prefix = buildQuestionPrefix(tok, q, maxLen, headMaxLen, order);
          const stAll = tok.encode(serializeState(state).split(tok.maskToken).join(" "));
          const actual = sequenceWithState(prefix, stAll, tok.sepId, maxLen, left);
          expect(actual).toEqual(expected);
          expect(prefix.nOptions).toBe(renderOptions(q).length);
        });
      }
    }
  }
});

describe("Agent state tokenization", () => {
  it("encodes the state text exactly once per systemOne call", async () => {
    const base = defaultTokenizer();
    const stateTexts: string[] = [];
    let encodeCalls = 0;
    const tok = {
      ...base,
      encode(text: string): number[] {
        encodeCalls++;
        stateTexts.push(text);
        return base.encode(text);
      },
    };
    const provider = {
      async runEncoder(batch: { inputIds: number[][] }) {
        return { lastHidden: batch.inputIds.map((row) => row.map(() => [1, 0])) };
      },
      async runHead(_h: unknown, batch: { markerPos: number[][] }) {
        return {
          logits: batch.markerPos.map((m) => m.map((_, i) => i + 1)),
          act: batch.markerPos.map(() => [3, 0]),
        };
      },
    };
    const agent = new Agent({ provider, tok } as never);
    const questions = {
      intent: { type: "choice", instructions: "?", criteria: { refund: "r", apology: "a", other: "o" } },
      angry: { type: "noul", instructions: "angry?" },
      csat: { type: "score", instructions: "rate", criteria: ["bad", "ok", "great"] },
    } as never;
    const state = { body: "charged twice, refund please" };
    const r = await agent.systemOne(state, questions);

    const serialized = serializeState(state);
    const stateEncodes = stateTexts.filter((t) => t === serialized).length;
    expect(stateEncodes).toBe(1); // not once per question
    // 1 state + per question (1 head + K options): 3 + 2 + 3 options/head sums = 1 + 4 + 3 + 4
    expect(encodeCalls).toBe(1 + (1 + 3) + (1 + 2) + (1 + 3));
    expect(Object.keys(r.answers).sort()).toEqual(["angry", "csat", "intent"]);
  });
});
