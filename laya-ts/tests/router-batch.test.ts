import { describe, expect, it } from "vitest";
import { Agent, Router } from "../src/index.js";
import type { BatchRequest } from "../src/index.js";

// TS mirror of tests/test_router_batch.py and the orchestration part of tests/test_batch.py.

const Q = { intent: { type: "noul", instructions: "Relevant?" } } as const;
const req = (state: unknown, overrides: Record<string, unknown> = {}): BatchRequest =>
  ({ state, questions: Q, ...overrides }) as unknown as BatchRequest;

const TYPED_KEYS = ["action", "needs_review", "outcome", "risk", "urgency"];
const typedQuestions = Object.fromEntries(
  TYPED_KEYS.map((k) => [k, { type: "noul", instructions: "?" }]),
);

// Fake checkpoint loader mirroring the Python fake_agent fixture: records which checkpoint
// was built and every predictBatch call, and fails when a state says "raise".
const makeRouter = (opts: Record<string, unknown> = {}) => {
  const built: string[] = [];
  const calls: [string, unknown[], Record<string, unknown>, unknown][] = [];
  const loader = (name: string, spec: { subfolder?: string | null }) => {
    const checkpoint = spec.subfolder ?? "english";
    built.push(checkpoint);
    return {
      checkpoint,
      async predictBatch(states: unknown[], questions: Record<string, unknown>, batchOpts?: unknown) {
        calls.push([checkpoint, [...states], questions, batchOpts ?? null]);
        if (states.includes("raise")) throw new Error("inference failed");
        return states.map((state) => ({
          model: "laya-rl-agent",
          answers: { seen: state },
          usage: {},
        }));
      },
      async systemOne(state: unknown, questions: Record<string, unknown>) {
        return (await this.predictBatch([state], questions))[0];
      },
    };
  };
  const router = new Router({ loader: loader as never, ...opts });
  return { router, built, calls };
};

const mixedItems = (): BatchRequest[] => [
  req("English text one"),
  req("\u0645\u0631\u062d\u0628\u0627"),
  req("English text two"),
  { state: "decision", questions: typedQuestions } as unknown as BatchRequest,
  req("forced", { model: "ml" }),
  req("forced english", { lang: "en" }),
  req("explicit task", { task: "typed_decisions" }),
];

describe("Router.routeBatch / predictBatch", () => {
  it("mixed groups keep input order and share one batch per checkpoint", async () => {
    const { router, built, calls } = makeRouter({ maxLoaded: 3, autoTaskDetection: true });
    const items = mixedItems();
    const decisions = router.routeBatch(items);
    expect(router.loaded).toEqual([]); // routeBatch routes without loading
    const results = await router.predictBatch(items);
    expect(results.map((r) => r.answers.seen)).toEqual(items.map((r) => r.state));
    expect(results.map((r) => r.routing)).toEqual(decisions);
    expect(built).toEqual(["english", "multilingual", "typed-decisions"]);
    expect(calls.map(([c, s]) => [c, s])).toEqual([
      ["english", ["English text one", "English text two", "forced english"]],
      ["multilingual", ["\u0645\u0631\u062d\u0628\u0627", "forced"]],
      ["typed-decisions", ["decision"]],
      ["typed-decisions", ["explicit task"]], // different schema: its own batch
    ]);
    expect(await router.predictMany([])).toEqual([]);
    expect(router.routeBatch([])).toEqual([]);
  });

  it("evicts least-recently-used checkpoints under capacity pressure", async () => {
    const { router, built } = makeRouter({ maxLoaded: 1, autoTaskDetection: true });
    await router.predictBatch(mixedItems());
    expect(built).toEqual(["english", "multilingual", "typed-decisions"]);
    expect(router.loaded).toEqual(["typed-decisions"]);
  });

  it("rejects malformed batches before loading anything", () => {
    const { router, built } = makeRouter();
    expect(() => router.routeBatch(null as never)).toThrow(TypeError);
    expect(() => router.routeBatch("text" as never)).toThrow(/requests must be an array/);
    expect(() => router.routeBatch([null] as never)).toThrow(/request 0/);
    expect(() => router.routeBatch([{ questions: Q }] as never)).toThrow(
      /request 0 is missing required key 'state'/,
    );
    expect(() => router.routeBatch([{ state: "x" }] as never)).toThrow(
      /request 0 is missing required key 'questions'/,
    );
    expect(() => router.routeBatch([req("x"), { state: "y", questions: null }] as never)).toThrow(
      /request 1 'questions'/,
    );
    expect(() => router.routeBatch([req("x"), req("y", { model: "invalid" })])).toThrow(
      /unknown model/,
    );
    expect(built).toEqual([]);
  });

  it("propagates inference failures and the router stays consistent afterwards", async () => {
    const { router, calls } = makeRouter({ maxLoaded: 2 });
    await expect(
      router.predictBatch([req("first"), req("raise", { lang: "ar" }), req("unreached", { lang: "ar" })]),
    ).rejects.toThrow("inference failed");
    expect(calls.map(([c, s]) => [c, s])).toEqual([
      ["english", ["first"]],
      ["multilingual", ["raise", "unreached"]], // one shared batch, even though it fails
    ]);
    const after = await router.predict("after failure", Q as never, { lang: "ar" });
    expect(after.answers.seen).toBe("after failure");
  });

  it("sends same-checkpoint same-schema requests in one agent batch, passing batchSize", async () => {
    const { router, calls } = makeRouter({ maxLoaded: 2 });
    const results = await router.predictBatch(
      [req("one", { model: "english" }), req("two", { model: "english" }), req("three", { model: "english" })],
      2,
    );
    expect(results.map((r) => r.answers.seen)).toEqual(["one", "two", "three"]);
    expect(calls.length).toBe(1);
    expect(calls[0][0]).toBe("english");
    expect(calls[0][1]).toEqual(["one", "two", "three"]);
    expect(calls[0][2]).toEqual(Q);
    expect(calls[0][3]).toEqual({ batchSize: 2 });
  });

  it("splits same-checkpoint requests with different question schemas", async () => {
    const { router, calls } = makeRouter({ maxLoaded: 2 });
    const q2 = { risk: { type: "noul", instructions: "Risky?" } };
    const results = await router.predictBatch([
      { state: "one", questions: Q, model: "english" } as unknown as BatchRequest,
      { state: "two", questions: q2, model: "english" } as unknown as BatchRequest,
      { state: "three", questions: Q, model: "english" } as unknown as BatchRequest,
    ]);
    expect(results.map((r) => r.answers.seen)).toEqual(["one", "two", "three"]);
    expect(calls.length).toBe(2);
    expect(calls[0][1]).toEqual(["one", "three"]);
    expect(calls[0][2]).toEqual(Q);
    expect(calls[1][1]).toEqual(["two"]);
    expect(calls[1][2]).toEqual(q2);
  });

  it("routeBatch forwards per-request langGuess overrides", () => {
    const { router, built } = makeRouter();
    const decisions = router.routeBatch([
      req("hola", { lang_guess: "es" }),
      req("hello", { langGuess: "en-US" }),
    ]);
    expect(decisions.map((d) => d.model)).toEqual(["multilingual", "english"]);
    expect(built).toEqual([]);
  });

  // A question schema that arrives with a different key order must be scored with its own
  // order: options are positional in the rendered sequence, so grouping reordered-but-equal
  // schemas would make the second request's batched answers differ from its single-request
  // answers. (TS mirror of the #166 regression test.)
  it("scores reordered-but-equal question schemas separately", async () => {
    const ordered = {
      intent: { type: "choice", instructions: "Pick one", criteria: { zulu: "last", alpha: "first" } },
    };
    const reordered = {
      intent: { type: "choice", instructions: "Pick one", criteria: { alpha: "first", zulu: "last" } },
    };
    const { router, calls } = makeRouter({ maxLoaded: 1 });
    const results = await router.predictBatch([
      { state: "one", questions: ordered } as unknown as BatchRequest,
      { state: "two", questions: reordered } as unknown as BatchRequest,
    ]);
    expect(results.length).toBe(2);
    expect(calls.map((c) => Object.keys((c[2] as never as typeof ordered).intent.criteria))).toEqual([
      ["zulu", "alpha"],
      ["alpha", "zulu"],
    ]);
  });

  it("runs router hooks per request and short-circuits skipped requests", async () => {
    const events: string[] = [];
    const { router, calls } = makeRouter({
      maxLoaded: 2,
      hooks: [{
        onPredictStart(ctx: never) {
          const c = ctx as { states: unknown[]; decision: { model: string }; skip: (r: unknown[]) => void };
          events.push(`start:${c.states[0]}:${c.decision.model}`);
          if (c.states[0] === "hit") {
            c.skip([{ model: "cached", answers: { seen: "cached" }, usage: {} }]);
          }
        },
        onPredictEnd(ctx: never) {
          const c = ctx as { states: unknown[]; results: { answers: { seen: unknown } }[] | null;
            usage: unknown; elapsedMs: number | null };
          events.push(`end:${c.states[0]}:${c.results?.[0]?.answers.seen}`);
          expect(c.usage).not.toBeNull(); // stamped before any end hook runs
          expect(c.elapsedMs).not.toBeNull();
        },
      }],
    });
    const results = await router.predictBatch([
      req("miss", { model: "english" }),
      req("hit", { model: "english" }),
      req("miss2", { model: "english" }),
    ]);
    expect(results.map((r) => r.answers.seen)).toEqual(["miss", "cached", "miss2"]);
    expect(calls.length).toBe(1); // the skipped request shared no forward pass
    expect(calls[0][1]).toEqual(["miss", "miss2"]);
    expect(events.filter((e) => e.startsWith("start:")).length).toBe(3);
    expect(events.filter((e) => e.startsWith("end:")).length).toBe(3);
    // predict still promises a routing key on a cache hit
    expect(results[1].routing.model).toBe("english");
  });

  it("fires onError per started request and end hooks even on failure", async () => {
    const errors: string[] = [];
    const ends: unknown[] = [];
    const { router } = makeRouter({
      maxLoaded: 2,
      hooks: [{
        onError(ctx: never) {
          const c = ctx as { states: unknown[]; error: Error };
          errors.push(`${c.states[0]}:${c.error.message}`);
        },
        onPredictEnd(ctx: never) {
          ends.push((ctx as { states: unknown[] }).states[0]);
        },
      }],
    });
    await expect(
      router.predictBatch([req("first"), req("raise", { lang: "ar" }), req("unreached", { lang: "ar" })]),
    ).rejects.toThrow("inference failed");
    // only the failed group's started-without-result requests see onError
    expect(errors).toEqual(["raise:inference failed", "unreached:inference failed"]);
    expect([...ends].sort()).toEqual(["first", "raise", "unreached"]);
  });

  it("splits groups when start hooks set different token-budget overrides", async () => {
    const { router, calls } = makeRouter({
      maxLoaded: 2,
      hooks: [{
        onPredictStart(ctx: never) {
          const c = ctx as { states: unknown[]; maxLen: number | null };
          if (c.states[0] === "two") c.maxLen = 64;
        },
      }],
    });
    await router.predictBatch([req("one", { model: "english" }), req("two", { model: "english" })]);
    expect(calls.map((c) => c[1])).toEqual([["one"], ["two"]]);
    // overrides are only passed when set, so Agent-like objects without them still work
    expect(calls[0][3]).toEqual({ batchSize: null });
    expect(calls[1][3]).toEqual({ batchSize: null, maxLen: 64 });
  });

  it("raises when the agent returns the wrong number of results", async () => {
    const router = new Router({
      loader: (() => ({
        async predictBatch() { return []; },
        async systemOne() { return {}; },
      })) as never,
    });
    await expect(
      router.predictBatch([req("a", { model: "english" }), req("b", { model: "english" })]),
    ).rejects.toThrow(/returned 0 results for 2 states/);
  });

  it("falls back to systemOne for agent-like objects without predictBatch", async () => {
    const router = new Router({
      loader: (() => ({
        async systemOne(state: unknown) {
          return { model: "stub", answers: { seen: state }, usage: {} };
        },
      })) as never,
    });
    const results = await router.predictBatch([req("a", { model: "english" }), req("b", { model: "english" })]);
    expect(results.map((r) => r.answers.seen)).toEqual(["a", "b"]);
    expect(results.every((r) => r.routing.model === "english")).toBe(true);
  });
});

// Agent-level batching, mirroring the orchestration tests in tests/test_batch.py: states
// sharing one question schema are collated into one forward pass per batchSize chunk.
const tokenizer = () => ({
  clsId: 1,
  sepId: 2,
  maskId: 3,
  padId: 0,
  maskToken: "[MASK]",
  encode(text: string): number[] {
    return Array.from(text).map((c) => c.codePointAt(0) ?? 0);
  },
});

const QUESTIONS = {
  flag: { type: "noul", instructions: "Relevant?" },
  pick: { type: "choice", instructions: "Pick", criteria: { a: "A", b: "B" } },
} as never;

// Row r gets logits [r + 1, 0], so each state's noul value reveals which row it decoded.
const makeCountingAgent = () => {
  const encodeCalls: number[] = [];
  const provider = {
    async runEncoder(batch: { inputIds: number[][] }) {
      encodeCalls.push(batch.inputIds.length);
      return { lastHidden: batch.inputIds.map(() => [[0, 0]]) };
    },
    async runHead(_hidden: unknown, batch: { inputIds: number[][] }) {
      const n = batch.inputIds.length;
      return {
        logits: Array.from({ length: n }, (_, r) => [r + 1, 0]),
        act: Array.from({ length: n }, () => [1, 0]),
      };
    },
  };
  const agent = new Agent({ provider, tok: tokenizer() } as never);
  return { agent, encodeCalls };
};

const expectedNoul = (row: number): number =>
  Math.round((1 / (Math.exp(row + 1) + 1)) * 1e4) / 1e4;

describe("Agent.predictBatch", () => {
  it("packs all states into one forward pass by default, in input order", async () => {
    const { agent, encodeCalls } = makeCountingAgent();
    const results = await agent.predictBatch(["one", "two", "three"], QUESTIONS);
    expect(encodeCalls).toEqual([6]); // 3 states x 2 questions, one pass
    expect(results.length).toBe(3);
    // state s decodes from rows 2s (flag) and 2s+1 (pick)
    expect(results.map((r) => (r.answers.flag as { noul: number }).noul)).toEqual([
      expectedNoul(0),
      expectedNoul(2),
      expectedNoul(4),
    ]);
  });

  it("batchSize chunks states across forward passes", async () => {
    const { agent, encodeCalls } = makeCountingAgent();
    const results = await agent.predictBatch(["1", "2", "3", "4", "5"], QUESTIONS, { batchSize: 2 });
    expect(encodeCalls).toEqual([4, 4, 2]); // 2+2+1 states x 2 questions
    expect(results.length).toBe(5);
    expect(results.map((r) => (r.answers.flag as { noul: number }).noul)).toEqual([
      expectedNoul(0), expectedNoul(2), expectedNoul(0), expectedNoul(2), expectedNoul(0),
    ]); // row offsets restart per chunk
  });

  it("handles empty states and empty questions without encoding", async () => {
    const { agent, encodeCalls } = makeCountingAgent();
    expect(await agent.predictBatch([], QUESTIONS)).toEqual([]);
    const [res] = await agent.predictBatch(["state"], {});
    expect(res.answers).toEqual({});
    expect(res.usage.input_tokens).toBe(0);
    expect(encodeCalls).toEqual([]);
  });

  it("matches systemOne output for a single state", async () => {
    const { agent } = makeCountingAgent();
    const batched = await agent.predictBatch(["state text"], QUESTIONS);
    const single = await agent.systemOne("state text", QUESTIONS);
    expect(batched[0]).toEqual(single);
  });

  it("reports per-state input token usage", async () => {
    const { agent } = makeCountingAgent();
    const results = await agent.predictBatch(["aaaa", "b"], QUESTIONS);
    expect(results[0].usage.input_tokens).toBeGreaterThan(results[1].usage.input_tokens);
    const solo = await agent.systemOne("aaaa", QUESTIONS);
    expect(results[0].usage.input_tokens).toBe(solo.usage.input_tokens);
  });

  it("supports skip from a start hook without encoding", async () => {
    const { agent, encodeCalls } = makeCountingAgent();
    const results = await agent.predictBatch(["a", "b"], QUESTIONS, {
      onPredictStart(ctx: never) {
        (ctx as { skip: (r: unknown[]) => void }).skip([
          { model: "cached", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } },
        ]);
      },
    });
    expect(results[0].model).toBe("cached");
    expect(encodeCalls).toEqual([]);
  });
});
