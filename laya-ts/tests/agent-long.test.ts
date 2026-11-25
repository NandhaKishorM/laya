import { afterEach, describe, expect, it, vi } from "vitest";
import { Agent, defaultTokenizer, type QuestionDef } from "../src/agent.js";
import { clearDefaultHooks, setDefaultHooks, type PredictContext } from "../src/hooks.js";
import { OptionStabilityHook } from "../src/option-stability.js";
import type { Batch, SessionProvider } from "../src/providers.js";

afterEach(() => { clearDefaultHooks(); vi.restoreAllMocks(); });
const questions: Record<string, QuestionDef> = { q: { type: "noul", instructions: "true?" } };
const document = "abcdefghijklmnopqrstuvwxyz".repeat(4);
const tok = { ...defaultTokenizer(),
  encode: (text: string) => Array.from(text, (c) => c.codePointAt(0)! + 200),
  decode: (ids: readonly number[]) => String.fromCodePoint(...ids.map((id) => id - 200)),
};
function makeAgent(increasing = false, options = {}) {
  const batches: Batch[] = [];
  let row = 0;
  const provider: SessionProvider = {
    async runEncoder(batch) { batches.push(batch); return { lastHidden: batch.inputIds.map(() => [[0]]) }; },
    async runHead(_hidden, batch) {
      return { logits: batch.qtype.map((_, i) => batch.markerPos[i].map((_, option) => option === 1 ? increasing ? ++row / 4 : 2 : 0)), act: batch.qtype.map(() => [0, 1]) };
    },
  };
  return { agent: new Agent({ provider, tok, max_len: 80, head_max_len: 32, ...options }), batches };
}

describe("Agent.predictLong", () => {
  it("preserves short-path output and supports encode-only custom tokenizers", async () => {
    const { agent } = makeAgent();
    const { decode: _decode, ...encodeOnly } = tok;
    agent.tok = encodeOnly;
    const ordinary = await agent.predict("short", questions);
    expect(await agent.predictLong("short", questions)).toEqual({ ...ordinary, usage: { ...ordinary.usage, windows: 1 } });
    await expect(agent.predictLong(document, questions, { window: 20 })).rejects.toThrow("needs decode");
  });

  it("rejects bad questions and unsafe options before hooks or inference", async () => {
    const { agent, batches } = makeAgent();
    const hook = vi.fn();
    await expect(agent.predictLong(document, { q: { type: "invalid" } }, { onPredictStart: hook })).rejects.toThrow('question "q"');
    for (const key of ["window", "stride", "batchSize"]) {
      for (const value of [0, -1, 1.1, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, true]) {
        await expect(agent.predictLong(document, questions, { [key]: value, onPredictStart: hook })).rejects.toThrow("positive safe integer");
      }
    }
    await expect(agent.predictLong(document, questions, { window: 10, stride: 11 })).rejects.toThrow("stride=11");
    await expect(agent.predictLong(document, questions, { aggregate: "mean" as never })).rejects.toThrow("only aggregate");
    expect(hook).not.toHaveBeenCalled();
    expect(batches).toHaveLength(0);
  });

  it("uses batched inference, earliest ties, candidate spans and numeric truncation counts", async () => {
    const { agent, batches } = makeAgent();
    const result = await agent.predictLong(document, questions, { window: 20, stride: 10, batchSize: 3 });
    expect(result.answers.q.window).toEqual({ index: 0, token_start: 0, token_end: 20, count: 10 });
    expect(result.usage.windows).toBe(10);
    expect(result.usage.truncated).toBe(0);
    expect(batches.map((b) => b.inputIds.length)).toEqual([3, 3, 3, 1]);
    expect(result.usage.output_tokens).toBe(0);
  });

  it("keeps inference answer identity through default/installed/per-call end-hook reordering", async () => {
    const events: string[] = [];
    setDefaultHooks({ onPredictStart: () => { events.push("default-start"); }, onPredictEnd: (ctx) => { ctx.results!.reverse(); events.push("default-end"); } });
    const { agent } = makeAgent(true, { hooks: { onPredictStart: () => { events.push("installed-start"); }, onPredictEnd: () => { events.push("installed-end"); } } });
    const result = await agent.predictLong(document, questions, { window: 20,
      onPredictStart: async () => { await Promise.resolve(); events.push("call-start"); },
      onPredictEnd: () => { events.push("call-end"); },
    });
    expect(result.answers.q.window).toEqual({ index: 9, token_start: 90, token_end: 104, count: 10 });
    expect(events).toEqual(["default-start", "installed-start", "call-start", "default-end", "installed-end", "call-end"]);
  });

  it("does not invent an origin for an answer object shared by inference rows", async () => {
    const { agent } = makeAgent();
    const base = await agent.predict("short", questions);
    vi.spyOn(agent as any, "_systemOneMany").mockImplementation(async (states: unknown[]) => states.map(() => ({ ...base })));
    const result = await agent.predictLong(document, questions, { window: 20 });
    expect(result.answers.q).not.toHaveProperty("window");
  });

  it("preserves the replacement-batch fallback but does not excuse missing added answers", async () => {
    const { agent } = makeAgent();
    const base = await agent.predict("short", questions);
    vi.spyOn(agent, "predictBatch").mockImplementation(async (states) => states.map(() => structuredClone(base)));
    const result = await agent.predictLong(document, questions, { window: 20 });
    expect(result.answers.q.window?.index).toBe(0);
    await expect(agent.predictLong(document, { ...questions, extra: questions.q }, { window: 20 })).rejects.toThrow('answer "extra" is missing');
  });

  it("clamps requested windows and adjusts their stride before inference, with a bounded default batch", async () => {
    const { agent, batches } = makeAgent();
    const warnings = vi.spyOn(console, "warn").mockImplementation(() => {});
    const wide = { q: { type: "choice", instructions: "pick", criteria: Array.from({ length: 11 }, (_, i) => `label${i}`) } };
    let actualStates: unknown[] = [];
    const result = await agent.predictLong("x".repeat(150), wide, { window: 80, stride: 40,
      onPredictStart(ctx) { actualStates = [...ctx.states]; },
    });
    const size = result.answers.q.window!.token_end;
    expect(size).toBeLessThan(40);
    expect(actualStates.length).toBe(result.usage.windows);
    expect(batches.length).toBeGreaterThan(1);
    expect(Math.max(...batches.map((b) => b.inputIds.length))).toBeLessThan(result.usage.windows);
    expect(warnings).toHaveBeenCalledTimes(2);
    await expect(agent.predictLong(document, wide, { window: 80, stride: 81 })).rejects.toThrow("stride=81");
  });

  it.each(["clone", "probabilities", "confidence", "shared"])("drops attribution for %s answer edits", async (edit) => {
    const { agent } = makeAgent();
    const result = await agent.predictLong(document, questions, { window: 20, onPredictEnd(ctx) {
      const first = ctx.results![0].answers as Record<string, any>;
      for (const row of ctx.results!) {
        const answers = row.answers as Record<string, any>;
        if (edit === "clone") answers.q = { ...answers.q };
        else if (edit === "probabilities") answers.q.probabilities = { yes: 1 };
        else if (edit === "confidence") answers.q.confidence = 0.123;
        else answers.q = first.q;
      }
    } });
    // End hooks that share the first unchanged answer retain that first answer's known
    // origin. Sharing at inference time, tested separately, has no unique origin.
    if (edit === "shared") expect(result.answers.q.window?.index).toBe(0);
    else expect(result.answers.q).not.toHaveProperty("window");
  });

  it("keeps attribution after action/reliability annotations and equal-value state replacements", async () => {
    const { agent } = makeAgent();
    const result = await agent.predictLong(document, questions, { window: 20,
      onPredictStart(ctx) { ctx.states = ctx.states.map((state) => `${state}`); },
      onPredictEnd(ctx) {
        for (const row of ctx.results!) {
          const answer = (row.answers as Record<string, any>).q;
          answer.action = { act_probability: 0.1 };
          answer.reliability = { note: "external" };
        }
      },
    });
    expect(result.answers.q.window?.index).toBe(0);
    expect(result.answers.q.action.act_probability).toBe(0.1);
  });

  it("omits all attribution when start hooks rewrite/reorder/remove windows", async () => {
    for (const rewrite of [(ctx: PredictContext) => { ctx.states.reverse(); },
      (ctx: PredictContext) => { ctx.states = ["replacement"]; },
      (ctx: PredictContext) => { ctx.states = []; }]) {
      const { agent } = makeAgent();
      vi.spyOn(console, "warn").mockImplementation(() => {});
      const result = await agent.predictLong(document, questions, { window: 20, onPredictStart: rewrite });
      if (!result.usage.windows) expect(result.answers).toEqual({});
      else expect(result.answers.q).not.toHaveProperty("window");
    }
  });

  it("accepts exactly one cached document, copies usage, and rejects malformed hook payloads", async () => {
    const { agent, batches } = makeAgent();
    const cached = await agent.predict("cached", questions);
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    const priorRuns = batches.length;
    for (const state of ["short", document]) {
      const result = await agent.predictLong(state, questions, { window: 20, onPredictStart: (ctx) => ctx.skip([cached as never]) });
      expect(result.usage.windows).toBe(0);
      expect(result).not.toBe(cached);
      expect(result.usage).not.toBe(cached.usage);
      expect(cached.usage).not.toHaveProperty("windows");
      await expect(agent.predictLong(state, questions, { window: 20, onPredictStart: (ctx) => ctx.skip([cached as never, cached as never]) })).rejects.toThrow("one result for the document");
      await expect(agent.predictLong(state, questions, { window: 20, onPredictStart: (ctx) => ctx.skip([{ model: "cache", answers: {}, usage: { input_tokens: "wrong", output_tokens: 0 } }]) })).rejects.toThrow("input_tokens");
    }
    expect(warning).toHaveBeenCalled();
    expect(batches).toHaveLength(priorRuns);
  });

  it("validates malformed result arrays and end-hook skip counts without masking the error", async () => {
    const { agent } = makeAgent();
    await expect(agent.predictLong("short", questions, { onPredictStart(ctx) { ctx.results = {} as never; } })).rejects.toThrow("array of results");
    const base = await agent.predict("short", questions);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await expect(agent.predictLong("short", questions, {
      onPredictStart(ctx) { ctx.skip([base as never]); },
      onPredictEnd(ctx) { ctx.results!.push(base as never); },
    })).rejects.toThrow("2 results");
    await expect(agent.predictLong(document, questions, { window: 20, onPredictEnd(ctx) {
      (ctx.results![0].answers as Record<string, any>).q.type = "choice";
      (ctx.results![0].answers as Record<string, any>).q.choice = "yes";
      (ctx.results![0].answers as Record<string, any>).q.probabilities = { yes: 1 };
    } })).rejects.toThrow("inconsistent types");
  });

  it("keeps uniformly rewritten answer schemas while applying the original aggregation rule", async () => {
    const { agent } = makeAgent();
    const result = await agent.predictLong(document, { q: { type: "choice", instructions: "pick", criteria: ["a", "b"] } }, {
      window: 20, onPredictEnd(ctx) {
        for (const row of ctx.results!) Object.assign((row.answers as Record<string, any>).q,
          { type: "score", score: 1, legend: { "0": "low", "1": "high" } });
      },
    });
    expect(result.answers.q.type).toBe("score");
    expect(result.answers.q).not.toHaveProperty("window");
  });

  it("runs the budget guard after inference even with suppressed hook errors", async () => {
    const { agent, batches } = makeAgent();
    await expect(agent.predictLong(document, questions, { window: 40, hooksRaise: false,
      onPredictStart(ctx) { ctx.maxLen = 45; },
    })).rejects.toThrow("scan was sized for 40");
    expect(batches.length).toBeGreaterThan(0);
    const short = makeAgent();
    await expect(short.agent.predictLong("x".repeat(40), questions, { hooksRaise: false,
      onPredictStart(ctx) { ctx.maxLen = 45; },
    })).rejects.toThrow("scan was sized");
    expect(short.batches.length).toBeGreaterThan(0);
  });

  it("rejects aggregate usage overflow after individually valid window reports", async () => {
    const { agent } = makeAgent();
    await expect(agent.predictLong(document, questions, { window: 20, onPredictEnd(ctx) {
      for (const result of ctx.results!) {
        (result.usage as Record<string, unknown>).input_tokens = Number.MAX_VALUE;
      }
    } })).rejects.toThrow("usage.input_tokens must be a finite non-negative number");
  });

  it("preserves own prototype-named keys in generic usage without changing its prototype", async () => {
    const { agent } = makeAgent();
    const result = await agent.predictLong(document, questions, { window: 20, onPredictEnd(ctx) {
      for (const row of ctx.results!) {
        row.usage = { ...(row.usage as object), ...JSON.parse('{"__proto__":2,"constructor":{"seen":true}}') };
      }
    } });
    expect(Object.hasOwn(result.usage, "__proto__")).toBe(true);
    expect(result.usage["__proto__"]).toBe(result.usage.windows * 2);
    expect(result.usage["constructor"]).toEqual({ seen: true });
    expect(Object.getPrototypeOf(result.usage)).toBe(Object.prototype);
  });

  it("detects nested in-place question edits against the pre-hook deep snapshot", async () => {
    const { agent, batches } = makeAgent();
    const q: Record<string, QuestionDef> = { q: { type: "choice", instructions: "pick", criteria: { a: "A", b: "B" } } };
    await expect(agent.predictLong(document, q, { window: 40, hooksRaise: false, onPredictStart(ctx) {
      Object.assign((ctx.questions.q as QuestionDef).criteria as object, Object.fromEntries(Array.from({ length: 8 }, (_, i) => [`extra${i}`, "extra"])));
    } })).rejects.toThrow("scan was sized");
    expect(batches.length).toBeGreaterThan(0);
  });

  it("allows only uniformly removed start-added questions", async () => {
    const { agent } = makeAgent();
    const start = (ctx: PredictContext) => { ctx.questions = { ...ctx.questions, probe: questions.q }; };
    const result = await agent.predictLong(document, questions, { window: 20, onPredictStart: start, onPredictEnd(ctx) {
      for (const row of ctx.results!) delete (row.answers as Record<string, unknown>).probe;
    } });
    expect(Object.keys(result.answers)).toEqual(["q"]);
    await expect(agent.predictLong(document, questions, { window: 20, onPredictStart: start, onPredictEnd(ctx) {
      delete (ctx.results![0].answers as Record<string, unknown>).probe;
    } })).rejects.toThrow('answer "probe" is missing from 1 of 10');
    await expect(agent.predictLong(document, questions, { window: 20, onPredictEnd(ctx) {
      for (const row of ctx.results!) delete (row.answers as Record<string, unknown>).q;
    } })).rejects.toThrow('answer "q" is missing from 10 of 10');
  });

  it("uses a real OptionStabilityHook without leaking probe questions or losing spans", async () => {
    const { agent } = makeAgent(false, { hooks: new OptionStabilityHook({ accept: null, escalate: null, variants: "fast", seed: "long" }) });
    const result = await agent.predictLong(document, { q: { type: "choice", instructions: "pick", criteria: ["a", "b"] } }, { window: 20 });
    expect(Object.keys(result.answers)).toEqual(["q"]);
    expect(result.answers.q.window?.index).toBe(0);
    expect(result.answers.q).toHaveProperty("reliability");
  });

  it("keeps concurrent long and ordinary contexts independent", async () => {
    const { agent } = makeAgent();
    const [left, right, ordinary] = await Promise.all([
      agent.predictLong(document, questions, { window: 20, onPredictStart: async (ctx) => { await Promise.resolve(); ctx.states.reverse(); } }),
      agent.predictLong(document, questions, { window: 30 }),
      agent.predict("short", questions),
    ]);
    expect(left.answers.q).not.toHaveProperty("window");
    expect(right.answers.q.window?.token_end).toBe(30);
    expect(ordinary.usage).not.toHaveProperty("windows");
  });

  it("sums numeric usage, shallow-merges dictionaries, and preserves last other fields", async () => {
    const { agent } = makeAgent();
    const result = await agent.predictLong("x".repeat(30), questions, { window: 20, stride: 20, onPredictEnd(ctx) {
      Object.assign(ctx.results![0].usage as object, { custom: 1, truncated: true, options: { first: { total: 2, distinct: 1, tokens_per_option: 4 } }, last: ["first"] });
      Object.assign(ctx.results![1].usage as object, { custom: 2, truncated: false, options: { second: { total: 3, distinct: 2, tokens_per_option: 4 } }, last: [] });
    } });
    expect(result.usage).toMatchObject({ custom: 3, truncated: 1, options: { first: { total: 2 }, second: { total: 3 } }, last: [], windows: 2 });
  });
});
