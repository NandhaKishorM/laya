import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import { buildSequence } from "../src/common.js";
import type { TokenizerLike } from "../src/tokenizer.js";

// TS mirror of tests/test_truncation.py (Python #181, issue #174): the state clamp must be
// reported from the token budget that actually applies, not guessed from character counts.

/** One token per whitespace-separated word, so token counts are exact and readable. */
const TOK: TokenizerLike = {
  clsId: 2,
  sepId: 3,
  maskId: 1,
  padId: 0,
  maskToken: "[MASK]",
  encode: (text: string) => text.split(/\s+/).filter(Boolean).map((w) => 10 + (w.length % 50)),
};

const Q = { t: "noul", ins: "Does the user ask for a refund?", crit: null } as const;
const LONG = Array.from({ length: 4000 }, (_, i) => `word${i}`).join(" ");

const statsFor = (state: string, maxLen = 512, headMaxLen = 192,
    truncateLeft = false, q: typeof Q | Record<string, unknown> = Q) =>
  buildSequence(TOK, state, q as never, maxLen, headMaxLen, undefined, truncateLeft).stats;

describe("sequence stats: the reported gap — a silent clamp (issue #174)", () => {
  it("a state that fits loses nothing, and says so", () => {
    const short = statsFor("the customer was billed twice");
    expect(short.truncated).toBe(false);
    expect(short.state_tokens_dropped).toBe(0);
    expect(short.state_tokens_used).toBe(short.state_tokens);
  });

  it("a state that does not fit reports the exact shortfall", () => {
    const clamped = statsFor(LONG);
    expect(clamped.truncated).toBe(true);
    expect(clamped.state_tokens).toBe(4000);
    expect(clamped.state_tokens_used).toBeLessThan(4000);
    expect(clamped.state_tokens_dropped).toBe(4000 - clamped.state_tokens_used);
    expect(clamped.state_tokens_used).toBeLessThanOrEqual(512);
  });
});

describe("the flag tracks the real window", () => {
  it("a 1024-token window keeps strictly more than a 512-token one", () => {
    const w512 = statsFor(LONG, 512);
    const w1024 = statsFor(LONG, 1024);
    expect(w1024.state_tokens_used).toBeGreaterThan(w512.state_tokens_used);
    expect(w1024.state_tokens_dropped).toBeLessThan(w512.state_tokens_dropped);
  });

  it("a mid-sized state is truncated on one checkpoint and intact on the other", () => {
    const mid = Array.from({ length: 600 }, (_, i) => `word${i}`).join(" ");
    expect(statsFor(mid, 512).truncated).toBe(true);
    expect(statsFor(mid, 1024).truncated).toBe(false);
  });

  it("a wider head leaves less room for the state", () => {
    const wide = {
      t: "choice",
      ins: "Which team should own this request, given the account tier and the contract terms?",
      crit: {
        billing: "invoices, payments, refunds, chargebacks and duplicate charges",
        technical: "bugs, outages, degraded performance and system errors",
        sales: "pricing, quotes, renewals and new contracts",
        other: "everything that does not belong to the three above",
      },
    };
    const narrow = { t: "noul", ins: "Refund?", crit: null };
    expect(statsFor(LONG, 512, 192, false, wide).state_tokens_used)
      .toBeLessThan(statsFor(LONG, 512, 192, false, narrow).state_tokens_used);
  });
});

describe("left truncation", () => {
  it("keeps as much as the right-hand clamp", () => {
    const left = statsFor(LONG, 512, 192, true);
    expect(left.truncated).toBe(true);
    expect(left.state_tokens_used).toBe(statsFor(LONG, 512).state_tokens_used);
  });

  it("with no room at all, slice(-0) is the whole state — count the final clamp instead", () => {
    const noRoom = statsFor(LONG, 8, 8, true);
    expect(noRoom.truncated).toBe(true);
    expect(noRoom.state_tokens_used).toBeLessThanOrEqual(8);
    expect(noRoom.state_tokens_dropped).toBe(noRoom.state_tokens - noRoom.state_tokens_used);
  });
});

describe("reported counts match the sequence", () => {
  it.each([
    ["512", 512, 192],
    ["1024", 1024, 512],
    ["tight", 64, 32],
  ] as const)("accounting/%s", (_name, maxLen, headMaxLen) => {
    const { ids, stats } = buildSequence(TOK, LONG, Q as never, maxLen, headMaxLen);
    expect(ids.length).toBeLessThanOrEqual(maxLen);
    expect(stats.state_tokens_used).toBeLessThanOrEqual(ids.length);
    expect(stats.state_tokens_used + stats.state_tokens_dropped).toBe(stats.state_tokens);
    expect(stats.truncated).toBe(stats.state_tokens_dropped > 0);
  });
});

describe("systemOne reports truncation in usage", () => {
  const provider = () => ({
    async runEncoder(b: { inputIds: number[][] }) {
      return { lastHidden: b.inputIds.map((row) => row.map(() => 0)) };
    },
    async runHead(_h: unknown, b: { inputIds: number[][] }) {
      return {
        // four columns: the widest question below has four options
        logits: b.inputIds.map(() => [0, 2, 0, 2]),
        act: b.inputIds.map(() => [1, 0]),
      };
    },
  });
  const agent = () =>
    new Agent({ provider: provider(), tok: TOK, max_len: 64, head_max_len: 48 } as never);
  const refund = { refund: { type: "noul", instructions: "Refund?" } } as never;

  it("publishes the flag, the dropped count and the affected questions", async () => {
    const long = Array.from({ length: 200 }, (_, i) => `w${i}`).join(" ");
    const r = await agent().systemOne(long, refund);
    expect(r.usage.input_tokens).toBeGreaterThan(0);
    expect(r.usage.output_tokens).toBe(0);
    expect(r.usage.state_tokens).toBe(200);
    expect(r.usage.truncated).toBe(true);
    expect(r.usage.state_tokens_dropped).toBeGreaterThan(0);
    expect((r.usage.state_tokens ?? 0) - (r.usage.state_tokens_dropped ?? 0))
      .toBeLessThanOrEqual(64);
    expect(r.usage.truncated_questions).toEqual(["refund"]);
  });

  it("a state that fits reports truncated: false", async () => {
    const r = await agent().systemOne("the customer was billed twice", refund);
    expect(r.usage.truncated).toBe(false);
    expect(r.usage.state_tokens_dropped).toBe(0);
    expect(r.usage.truncated_questions).toEqual([]);
  });

  it("questions over one state can disagree — only the starved head is named", async () => {
    const wideIns = "Which team should own this request given the account tier and contract terms?";
    const fortyWords = Array.from({ length: 40 }, (_, i) => `w${i}`).join(" ");
    const r = await agent().systemOne(fortyWords, {
      wide: { type: "choice", instructions: wideIns,
              criteria: { billing: "invoices and payments", technical: "bugs and outages",
                          sales: "pricing and quotes", other: "everything else here" } },
      narrow: { type: "noul", instructions: "Refund?" },
    } as never);
    expect(r.usage.truncated).toBe(true);
    expect(r.usage.truncated_questions).toEqual(["wide"]);
  });

  it("predictBatch reports per state, aligned by index", async () => {
    const long = Array.from({ length: 200 }, (_, i) => `w${i}`).join(" ");
    const out = await agent().predictBatch(["short state here", long], refund);
    expect(out[0].usage.truncated).toBe(false);
    expect(out[1].usage.truncated).toBe(true);
    expect(out[1].usage.state_tokens).toBe(200);
  });

  it("empty questions keep the bare usage shape (no state ever encoded)", async () => {
    const r = await agent().systemOne("hi", {});
    expect(r.usage).toEqual({ input_tokens: 0, output_tokens: 0 });
  });
});

describe("array states are left-truncated", () => {
  it("the encoder row is the tail of the state, as in Python", async () => {
    const rows: number[][] = [];
    const provider = {
      async runEncoder(b: { inputIds: number[][] }) {
        rows.push(...b.inputIds);
        return { lastHidden: b.inputIds.map((row) => row.map(() => 0)) };
      },
      async runHead(_h: unknown, b: { inputIds: number[][] }) {
        return { logits: b.inputIds.map(() => [0, 2]), act: b.inputIds.map(() => [1, 0]) };
      },
    };
    const agent = () =>
      new Agent({ provider, tok: TOK, max_len: 64, head_max_len: 48 } as never);
    const state = Array.from({ length: 200 }, (_, i) => `w${i}`);
    const q = { t: "noul", ins: "Refund?", crit: undefined } as const;
    await agent().systemOne(state, { refund: { type: "noul", instructions: "Refund?" } } as never);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toEqual(buildSequence(TOK, state, q as never, 64, 48, undefined, true).ids);
    expect(rows[0]).not.toEqual(buildSequence(TOK, state, q as never, 64, 48).ids);
  });
});
