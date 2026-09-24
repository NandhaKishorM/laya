import { describe, expect, it } from "vitest";
import { Router, normaliseName, _questionSchema, type RouterBatchRequest } from "../src/router.js";
import type { QuestionDef, SystemOneResult } from "../src/agent.js";
import type { PredictContext } from "../src/hooks.js";

const Q: Record<string, QuestionDef> = {
  intent: { type: "noul", instructions: "Relevant?" },
};

function req(state: unknown, overrides: Partial<RouterBatchRequest> = {}): RouterBatchRequest {
  return { state, questions: Q, ...overrides };
}

class BatchFakeAgent {
  checkpoint: string;
  calls: Array<{ states: unknown[]; questions: Record<string, QuestionDef>; opts?: Record<string, unknown> }> = [];

  constructor(checkpoint: string) {
    this.checkpoint = checkpoint;
  }

  async predictBatch(
    states: unknown[],
    questions: Record<string, QuestionDef>,
    opts?: Record<string, unknown>,
  ): Promise<SystemOneResult[]> {
    this.calls.push({ states: [...states], questions, opts: opts ? { ...opts } : undefined });
    if (states.includes("boom")) {
      throw new Error("inference failed");
    }
    return states.map((s) => ({
      model: "fake",
      answers: { seen: s } as any,
      usage: { input_tokens: typeof s === "string" ? s.length : 1, output_tokens: 0 },
    }));
  }

  async systemOne(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts?: Record<string, unknown>,
  ): Promise<SystemOneResult> {
    const res = await this.predictBatch([state], questions, opts);
    return res[0];
  }
}

function makeRouterWithLoader(capacity = 2, autoTaskDetection = false) {
  const agents: Record<string, BatchFakeAgent> = {
    english: new BatchFakeAgent("english"),
    multilingual: new BatchFakeAgent("multilingual"),
    "typed-decisions": new BatchFakeAgent("typed-decisions"),
  };
  const built: string[] = [];

  const router = new Router({
    maxLoaded: capacity,
    autoTaskDetection,
    loader: async (name) => {
      built.push(name);
      return agents[name];
    },
  });

  return { router, agents, built };
}

describe("router", () => {
  it("normalises aliases", () => { expect(normaliseName("en")).toBe("english"); });
  it("rejects unknown model", () => { expect(() => normaliseName("jev-1")).toThrow(); });
  it("routes hindi script to multilingual without loading", () => {
    const r = new Router();
    expect(r.route({ body: "मुझसे दो बार शुल्क लिया गया" }, {}).model).toBe("multilingual");
    expect(r.loaded).toEqual([]);
  });
  it("explicit model wins", () => {
    expect(new Router().route("hi", {}, { model: "typed-decisions" }).model).toBe("typed-decisions");
  });
  it("lang_guess callable routes", () => {
    const r = new Router();
    expect(r.route("text", {}, { langGuess: () => "ro" }).model).toBe("multilingual");
  });
});

describe("router batching validation and routing", () => {
  it("rejects invalid requests argument", async () => {
    const { router } = makeRouterWithLoader();
    expect(() => router.routeBatch(null as any)).toThrow(TypeError);
    expect(() => router.routeBatch({} as any)).toThrow("requests must be a sequence");
    expect(() => router.routeBatch("text" as any)).toThrow("requests must be a sequence");
    await expect(router.predictBatch(null as any)).rejects.toThrow("requests must be a sequence");
  });

  it("rejects malformed items before loading", async () => {
    const { router, built } = makeRouterWithLoader();
    expect(() => router.routeBatch([null as any])).toThrow("request 0 must be a dict");
    expect(() => router.routeBatch([{ questions: Q } as any])).toThrow("request 0 is missing required key 'state'");
    expect(() => router.routeBatch([{ state: "x" } as any])).toThrow("request 0 is missing required key 'questions'");
    expect(() => router.routeBatch([req("x"), { state: "y", questions: null } as any])).toThrow("request 1 'questions' must be a dict");
    expect(() => router.routeBatch([req("x"), req("y", { model: "invalid" })])).toThrow("unknown model");

    await expect(router.predictBatch([null as any])).rejects.toThrow("request 0 must be a dict");
    expect(built).toEqual([]);
    expect(router.loaded).toEqual([]);
  });

  it("handles empty batch", async () => {
    const { router, built } = makeRouterWithLoader();
    expect(router.routeBatch([])).toEqual([]);
    expect(await router.predictBatch([])).toEqual([]);
    expect(built).toEqual([]);
    expect(router.loaded).toEqual([]);
  });

  it("routeBatch forwards langGuess without loading checkpoints", () => {
    const { router, built } = makeRouterWithLoader();
    const decisions = router.routeBatch([
      req("hola", { langGuess: "es" }),
      req("hello", { lang_guess: "en-US" }),
    ]);
    expect(decisions.map((d) => d.model)).toEqual(["multilingual", "english"]);
    expect(built).toEqual([]);
    expect(router.loaded).toEqual([]);
  });
});

describe("router predictBatch execution, ordering, and LRU", () => {
  it.each([
    { capacity: 1, expectedBuilt: ["english", "multilingual", "typed-decisions"], expectedLoaded: ["typed-decisions"] },
    { capacity: 2, expectedBuilt: ["english", "multilingual", "typed-decisions"], expectedLoaded: ["multilingual", "typed-decisions"] },
    { capacity: 3, expectedBuilt: ["english", "multilingual", "typed-decisions"], expectedLoaded: ["english", "multilingual", "typed-decisions"] },
  ])("mixed groups keep original order and respect LRU (capacity=$capacity)", async ({ capacity, expectedBuilt, expectedLoaded }) => {
    const { router, agents, built } = makeRouterWithLoader(capacity, true);
    const typedQuestions = {
      action: { type: "noul", instructions: "?" },
      needs_review: { type: "noul", instructions: "?" },
      outcome: { type: "noul", instructions: "?" },
      risk: { type: "noul", instructions: "?" },
      urgency: { type: "noul", instructions: "?" },
    };

    const items: RouterBatchRequest[] = [
      req("English text one"),
      req("مرحبا"),
      req("English text two"),
      { state: "decision", questions: typedQuestions },
      req("forced", { model: "ml" }),
      req("forced english", { lang: "en" }),
      req("explicit task", { task: "typed_decisions" }),
    ];

    const decisions = router.routeBatch(items);
    expect(router.loaded).toEqual([]);

    const results = await router.predictBatch(items);
    expect(results.length).toBe(items.length);
    expect(results.map((r) => (r.answers as any).seen)).toEqual(items.map((i) => i.state));
    expect(results.map((r) => r.routing)).toEqual(decisions);
    expect(built).toEqual(expectedBuilt);
    expect(router.loaded).toEqual(expectedLoaded);

    expect(agents.english.calls.map((c) => c.states)).toEqual([
      ["English text one", "English text two", "forced english"],
    ]);
    expect(agents.multilingual.calls.map((c) => c.states)).toEqual([
      ["مرحبا", "forced"],
    ]);
    expect(agents["typed-decisions"].calls.map((c) => c.states)).toEqual([
      ["decision"],
      ["explicit task"],
    ]);
  });

  it("same checkpoint and same questions coalesce into one agent call with batchSize", async () => {
    const { router, agents } = makeRouterWithLoader(2);
    const items = [
      req("one", { model: "english" }),
      req("two", { model: "english" }),
      req("three", { model: "english" }),
    ];

    const results = await router.predictBatch(items, { batchSize: 2 });
    expect(results.map((r) => (r.answers as any).seen)).toEqual(["one", "two", "three"]);
    expect(agents.english.calls.length).toBe(1);
    expect(agents.english.calls[0].states).toEqual(["one", "two", "three"]);
    expect(agents.english.calls[0].opts).toEqual({ batchSize: 2 });
  });

  it("forwards numeric batchSize parameter directly to agent.predictBatch", async () => {
    const { router, agents } = makeRouterWithLoader(2);
    const items = [
      req("one", { model: "english" }),
      req("two", { model: "english" }),
      req("three", { model: "english" }),
    ];

    const results = await router.predictBatch(items, 2);
    expect(results.map((r) => (r.answers as any).seen)).toEqual(["one", "two", "three"]);
    expect(agents.english.calls.length).toBe(1);
    expect(agents.english.calls[0].states).toEqual(["one", "two", "three"]);
    expect(agents.english.calls[0].opts).toEqual({ batchSize: 2 });
  });

  it("same checkpoint and different questions split agent batches", async () => {
    const { router, agents } = makeRouterWithLoader(2);
    const q2 = { risk: { type: "noul", instructions: "Risky?" } };
    const items = [
      req("one", { model: "english" }),
      req("two", { model: "english", questions: q2 }),
      req("three", { model: "english" }),
    ];

    const results = await router.predictBatch(items);
    expect(results.map((r) => (r.answers as any).seen)).toEqual(["one", "two", "three"]);
    expect(agents.english.calls.length).toBe(2);
    expect(agents.english.calls[0].states).toEqual(["one", "three"]);
    expect(agents.english.calls[0].questions).toBe(Q);
    expect(agents.english.calls[1].states).toEqual(["two"]);
    expect(agents.english.calls[1].questions).toBe(q2);
  });

  it("equal questions with different option order score separately (#166)", async () => {
    const { router, agents } = makeRouterWithLoader(1);
    const ordered = {
      intent: { type: "choice", instructions: "Pick", criteria: { zulu: "last", alpha: "first" } },
    };
    const reordered = {
      intent: { type: "choice", instructions: "Pick", criteria: { alpha: "first", zulu: "last" } },
    };

    expect(_questionSchema(ordered)).not.toBe(_questionSchema(reordered));

    const items: RouterBatchRequest[] = [
      { state: "one", questions: ordered, model: "english" },
      { state: "two", questions: reordered, model: "english" },
    ];

    const results = await router.predictBatch(items);
    expect(results.length).toBe(2);
    expect(agents.english.calls.length).toBe(2);
    expect(Object.keys(agents.english.calls[0].questions.intent.criteria as any)).toEqual(["zulu", "alpha"]);
    expect(Object.keys(agents.english.calls[1].questions.intent.criteria as any)).toEqual(["alpha", "zulu"]);
  });
});

describe("router batch hooks and error handling", () => {
  it("partial ctx.skip() bypasses agent, preserves routing, and other requests infer normally", async () => {
    const cachedX = {
      model: "cached-model",
      answers: { seen: "cached-x" } as any,
      usage: { input_tokens: 0, output_tokens: 0 },
    };
    const { router, agents } = makeRouterWithLoader();
    router.addHook({
      onPredictStart(ctx: PredictContext) {
        if (ctx.states[0] === "x") {
          ctx.skip([cachedX]);
        }
      },
    });

    const results = await router.predictBatch([
      req("x", { model: "english" }),
      req("y", { model: "english" }),
    ]);

    expect(agents.english.calls.length).toBe(1);
    expect(agents.english.calls[0].states).toEqual(["y"]);

    expect((results[0].answers as any).seen).toBe("cached-x");
    expect(results[0].routing.model).toBe("english");
    expect((results[1].answers as any).seen).toBe("y");
    expect(results[1].routing.model).toBe("english");
  });

  it("onPredictStart state rewrite reaches the agent", async () => {
    const { router, agents } = makeRouterWithLoader();
    router.addHook({
      onPredictStart(ctx: PredictContext) {
        ctx.states = [String(ctx.states[0]).replace("secret", "[redacted]")];
      },
    });

    const out = await router.predictBatch([
      req("a secret", { model: "english" }),
      req("b secret", { model: "multilingual" }),
      req("c secret", { model: "english" }),
    ]);

    expect(agents.english.calls[0].states).toEqual(["a [redacted]", "c [redacted]"]);
    expect(agents.multilingual.calls[0].states).toEqual(["b [redacted]"]);
    expect(out.map((o) => (o.answers as any).seen)).toEqual(["a [redacted]", "b [redacted]", "c [redacted]"]);
  });

  it("onPredictStart question mutation creates a separate question group", async () => {
    const { router, agents } = makeRouterWithLoader();
    const qOther = { other: { type: "noul", instructions: "Other?" } };
    router.addHook({
      onPredictStart(ctx: PredictContext) {
        if (ctx.states[0] === "b") {
          ctx.questions = qOther;
        }
      },
    });

    await router.predictBatch([
      req("a", { model: "english" }),
      req("b", { model: "english" }),
      req("c", { model: "english" }),
    ]);

    expect(agents.english.calls.length).toBe(2);
    expect(agents.english.calls[0].states).toEqual(["a", "c"]);
    expect(agents.english.calls[0].questions).toBe(Q);
    expect(agents.english.calls[1].states).toEqual(["b"]);
    expect(agents.english.calls[1].questions).toBe(qOther);
  });

  it("token budget override creates a separate question group and passes opts", async () => {
    const { router, agents } = makeRouterWithLoader();
    router.addHook({
      onPredictStart(ctx: PredictContext) {
        if (ctx.states[0] === "long") {
          ctx.maxLen = 1024;
        }
      },
    });

    await router.predictBatch([
      req("short", { model: "english" }),
      req("long", { model: "english" }),
      req("short again", { model: "english" }),
    ]);

    expect(agents.english.calls.length).toBe(2);
    expect(agents.english.calls[0].states).toEqual(["short", "short again"]);
    expect(agents.english.calls[0].opts).toBeUndefined();
    expect(agents.english.calls[1].states).toEqual(["long"]);
    expect(agents.english.calls[1].opts).toEqual({ maxLen: 1024 });
  });

  it("failure dispatches onError only to unskipped requests, and onPredictEnd to all started requests", async () => {
    const events: Array<{ type: string; state: unknown }> = [];
    let cachedEndedHasError: unknown = null;
    let cachedEndedHasResults: unknown = null;

    const cached = {
      model: "cached",
      answers: { seen: "cached" } as any,
      usage: { input_tokens: 0, output_tokens: 0 },
    };

    const { router } = makeRouterWithLoader();
    router.addHook({
      onPredictStart(ctx: PredictContext) {
        events.push({ type: "start", state: ctx.states[0] });
        if (ctx.states[0] === "cached") {
          ctx.skip([cached]);
        }
      },
      onError(ctx: PredictContext) {
        events.push({ type: "error", state: ctx.states[0] });
      },
      onPredictEnd(ctx: PredictContext) {
        events.push({ type: "end", state: ctx.states[0] });
        if (ctx.states[0] === "cached") {
          cachedEndedHasError = ctx.error;
          cachedEndedHasResults = ctx.results;
        }
      },
    });

    await expect(
      router.predictBatch([
        req("ok", { model: "multilingual" }),
        req("cached", { model: "english" }),
        req("boom", { model: "english" }),
      ]),
    ).rejects.toThrow("inference failed");

    // The first group (multilingual with "ok") settled cleanly
    expect(events.filter((e) => e.state === "ok").map((e) => e.type)).toEqual(["start", "end"]);

    // In the failing group:
    // "cached" started, was skipped, was NOT errored, and ended cleanly
    expect(events.filter((e) => e.state === "cached").map((e) => e.type)).toEqual(["start", "end"]);
    expect(cachedEndedHasError).toBeNull();
    expect(cachedEndedHasResults).toBeDefined();

    // "boom" started, errored, and ended
    expect(events.filter((e) => e.state === "boom").map((e) => e.type)).toEqual(["start", "error", "end"]);
  });

  it("elapsedMs and usage are set for all contexts before any onPredictEnd runs", async () => {
    const groupCtxs: PredictContext[] = [];
    const elapsedWhenFirstEndRan: boolean[][] = [];

    const { router } = makeRouterWithLoader();
    router.addHook({
      onPredictStart(ctx: PredictContext) {
        groupCtxs.push(ctx);
      },
      onPredictEnd() {
        if (elapsedWhenFirstEndRan.length === 0) {
          elapsedWhenFirstEndRan.push(groupCtxs.map((c) => c.elapsedMs !== null));
        }
      },
    });

    await router.predictBatch([
      req("a", { model: "english" }),
      req("b", { model: "english" }),
      req("c", { model: "english" }),
    ]);

    expect(elapsedWhenFirstEndRan).toEqual([[true, true, true]]);
  });

  it("onRoute hook may replace decision with a plain dict", async () => {
    const { router, agents } = makeRouterWithLoader();
    router.addHook({
      onRoute(ctx: PredictContext) {
        ctx.decision = { ...ctx.decision, model: "multilingual" };
      },
    });

    const out = await router.predictBatch([req("pinned", { model: "english" })]);
    expect(agents.multilingual.calls[0].states).toEqual(["pinned"]);
    expect(out[0].routing.model).toBe("multilingual");
  });
});
