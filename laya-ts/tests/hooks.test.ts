import { describe, expect, it, vi } from "vitest";
import { Agent, defaultTokenizer } from "../src/agent.js";
import { Router } from "../src/router.js";
import {
  BaseHook,
  HookRegistry,
  addDefaultHook,
  clearDefaultHooks,
  defaultHooks,
  normaliseHooks,
  setDefaultHooks,
  type Hook,
  type PredictContext,
} from "../src/hooks.js";

const QUESTIONS = { q: { type: "noul", instructions: "?" } } as never;

/** A real Agent with the model replaced by a recording fake provider. */
function makeAgent(seen: string[] = [], opts: Record<string, unknown> = {}) {
  const provider = {
    async runEncoder(_b: unknown) {
      return { lastHidden: [[1, 0], [0, 1]] };
    },
    async runHead(_h: unknown) {
      return { logits: [[2, 0]], act: [[3, 0]] };
    },
  };
  const tok = {
    ...defaultTokenizer(),
    encode(text: string): number[] {
      seen.push(text);
      return text.split(/\s+/).filter(Boolean).map(() => 7);
    },
  };
  return new Agent({ provider, tok, ...opts } as never);
}

function makeRouter(extra: Record<string, unknown> = {}) {
  const calls: unknown[] = [];
  const loader = (_name: string) => ({
    async systemOne(state: unknown, _q: unknown) {
      calls.push(state);
      return { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 5, output_tokens: 0 } };
    },
  });
  const router = new Router({ loader, ...extra } as never);
  return { router, calls };
}

describe("agent hooks", () => {
  it("start and end hooks fire once each with a shared context", async () => {
    const events: string[] = [];
    let startCtx: PredictContext | null = null;
    const hook: Hook = {
      onPredictStart(ctx) {
        events.push("start");
        startCtx = ctx;
      },
      onPredictEnd(ctx) {
        events.push("end");
        expect(ctx.runId).toBe((startCtx as PredictContext | null)?.runId);
        expect(ctx.results).not.toBeNull();
        expect(ctx.elapsedMs).not.toBeNull();
        expect(ctx.usage?.input_tokens).toBeGreaterThan(0);
        expect(ctx.agent).toBe(agent);
      },
    };
    const agent = makeAgent([], { hooks: [hook] });
    const r = await agent.systemOne("hi", QUESTIONS);
    expect(events).toEqual(["start", "end"]);
    expect(r.answers).toHaveProperty("q");
  });

  it("a start hook may rewrite the state the model sees", async () => {
    const seen: string[] = [];
    const agent = makeAgent(seen);
    await agent.systemOne("original text", QUESTIONS, {
      onPredictStart(ctx) {
        ctx.states = ["rewritten text"];
      },
    });
    expect(seen.join(" ")).toContain("rewritten text");
    expect(seen.join(" ")).not.toContain("original text");
  });

  it("ctx.skip() short-circuits inference; end hooks still run", async () => {
    const seen: string[] = [];
    const events: string[] = [];
    const cached = { model: "cache", answers: { q: { type: "noul", noul: 1 } }, usage: { input_tokens: 0, output_tokens: 0 } };
    const agent = makeAgent(seen, {
      onPredictStart(ctx) {
        events.push("start");
        ctx.skip([cached]);
      },
      onPredictEnd(ctx) {
        events.push("end");
        expect(ctx.usage?.input_tokens).toBe(0);
      },
    });
    const r = await agent.systemOne("hi", QUESTIONS);
    expect(events).toEqual(["start", "end"]);
    expect(seen).toEqual([]); // the tokenizer never ran
    expect(r).toEqual(cached);
  });

  it("an end hook may rewrite the results", async () => {
    const agent = makeAgent([], {
      onPredictEnd(ctx) {
        for (const r of ctx.results ?? []) r["stamped"] = true;
      },
    });
    const r = (await agent.systemOne("hi", QUESTIONS)) as Record<string, unknown>;
    expect(r["stamped"]).toBe(true);
  });

  it("per-call hooks run after installed hooks", async () => {
    const order: string[] = [];
    const agent = makeAgent([], { hooks: [{ onPredictStart: () => order.push("installed") }] });
    await agent.systemOne("hi", QUESTIONS, { onPredictStart: () => order.push("per-call") });
    expect(order).toEqual(["installed", "per-call"]);
  });

  it("hooksRaise=false warns and continues when a hook fails", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const agent = makeAgent([], {
      hooksRaise: false,
      onPredictStart() {
        throw new Error("telemetry down");
      },
    });
    const r = await agent.systemOne("hi", QUESTIONS);
    expect(r.answers).toHaveProperty("q");
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("telemetry down"));
    warn.mockRestore();
  });

  it("onError fires on inference failure; the original error propagates and end hooks still run", async () => {
    const events: string[] = [];
    const agent = makeAgent([], {
      hooks: [
        {
          onError(ctx) {
            events.push("error");
            expect(String(ctx.error)).toContain("question");
          },
          onPredictEnd() {
            events.push("end");
          },
        },
      ],
    });
    await expect(agent.systemOne("hi", { bad: { type: "nope" } } as never)).rejects.toThrow(
      'question "bad"',
    );
    expect(events).toEqual(["error", "end"]);
  });

  it("a failing onError hook does not mask the original failure", async () => {
    const agent = makeAgent([], {
      hooks: [
        {
          onError() {
            throw new Error("hook broke too");
          },
        },
      ],
    });
    await expect(agent.systemOne("hi", { bad: { type: "nope" } } as never)).rejects.toThrow(
      'question "bad"',
    );
  });

  it("a failing end hook propagates on the success path", async () => {
    const agent = makeAgent([], {
      onPredictEnd() {
        throw new Error("end hook broke");
      },
    });
    await expect(agent.systemOne("hi", QUESTIONS)).rejects.toThrow("end hook broke");
  });
});

describe("router hooks", () => {
  it("onRoute observes and may replace the decision", async () => {
    const seen: string[] = [];
    const { router, calls } = makeRouter({
      hooks: [
        {
          onRoute(ctx: PredictContext) {
            seen.push(ctx.decision?.model as string);
            ctx.decision = {
              model: "multilingual",
              repo: "convaiinnovations/laya/multilingual",
              reason: "pinned by a hook",
              detection: null,
              workflow: null,
            };
          },
        },
      ],
    });
    const d = router.route("hello there", QUESTIONS);
    expect(seen).toEqual(["english"]); // detection ran before the hook replaced it
    expect(d.model).toBe("multilingual");
    const r = await router.predict("hello there", QUESTIONS);
    expect(r.routing.model).toBe("multilingual");
    expect(calls).toEqual(["hello there"]);
  });

  it("predict wraps route+infer: start hooks see ctx.decision, end hooks see results and usage", async () => {
    let startCtx: PredictContext | null = null;
    const { router } = makeRouter({
      onPredictStart(ctx: PredictContext) {
        startCtx = ctx;
        expect(ctx.decision?.model).toBe("english");
        expect(ctx.model).toBe("english");
        expect(ctx.router).toBe(router);
      },
      onPredictEnd(ctx: PredictContext) {
        expect(ctx.runId).toBe((startCtx as PredictContext | null)?.runId);
        expect(ctx.results?.[0]).toHaveProperty("routing");
        expect(ctx.usage).toEqual({ input_tokens: 5, output_tokens: 0 });
      },
    });
    const r = await router.predict("plain english text", QUESTIONS);
    expect(r.routing.model).toBe("english");
  });

  it("a cache hit still gets a routing key, without overwriting an existing one", async () => {
    const { router, calls } = makeRouter({
      onPredictStart(ctx: PredictContext) {
        ctx.skip([{ model: "cache", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } }]);
      },
    });
    const r = await router.predict("plain english text", QUESTIONS);
    expect(calls).toEqual([]); // inference skipped
    expect(r.routing.model).toBe("english");

    const { router: router2 } = makeRouter({
      onPredictStart(ctx: PredictContext) {
        ctx.skip([
          {
            model: "cache",
            answers: {},
            usage: { input_tokens: 0, output_tokens: 0 },
            routing: { model: "typed-decisions" },
          },
        ]);
      },
    });
    const r2 = await router2.predict("plain english text", QUESTIONS);
    expect(r2.routing.model).toBe("typed-decisions"); // cached routing wins
  });

  it("onLoad fires on the first load only; onEvict fires for the LRU victim", async () => {
    const events: string[] = [];
    const { router } = makeRouter({
      maxLoaded: 1,
      hooks: [
        {
          onLoad: (ctx: PredictContext) => events.push(`load:${String(ctx.model)}`),
          onEvict: (ctx: PredictContext) => events.push(`evict:${String(ctx.model)}`),
        },
      ],
    });
    await router.load("english");
    await router.load("english"); // cache hit: no event
    await router.load("multilingual"); // exceeds maxLoaded=1
    expect(events).toEqual(["load:english", "evict:english", "load:multilingual"]);
    expect(router.loaded).toEqual(["multilingual"]);
  });
});

describe("hook registry and normalisation", () => {
  it("a plain callable as hooks is a TypeError; as onPredictStart it adapts", () => {
    const fn = () => {};
    expect(() => normaliseHooks(fn as never)).toThrow(/ambiguous/);
    const out = normaliseHooks(undefined, fn);
    expect(out).toHaveLength(1);
    expect(typeof out[0].onPredictStart).toBe("function");
  });

  it("rejects the same callable for both start and end", () => {
    const fn = () => {};
    expect(() => normaliseHooks(undefined, fn, fn)).toThrow(/different callables/);
  });

  it("rejects objects with no hook methods and non-callable onPredictStart", () => {
    expect(() => normaliseHooks({ notAHook: 1 } as never)).toThrow(/at least one hook method/);
    expect(() => normaliseHooks(undefined, 42 as never)).toThrow(/onPredictStart must be callable/);
  });

  it("addHook/removeHook chain and withHooks scopes async callbacks", async () => {
    const reg = new HookRegistry();
    const hits: string[] = [];
    const hook: Hook = { onPredictStart: () => hits.push("hit") };
    expect(reg.addHook(hook)).toBe(reg);
    expect(reg.hooks).toHaveLength(1);
    expect(reg.removeHook(hook)).toBe(true);
    expect(reg.removeHook(hook)).toBe(false);

    const { router } = makeRouter();
    await router.withHooks([hook], async () => {
      expect(router.hooks).toHaveLength(1);
      await router.predict("plain english text", QUESTIONS);
    });
    expect(hits).toEqual(["hit"]);
    expect(router.hooks).toHaveLength(0); // removed after the callback settles
  });
});

describe("BaseHook and process-wide default hooks (py #276 parity)", () => {
  it("BaseHook/overridden event fires; a no-op instance is harmless", async () => {
    const log: string[] = [];
    class OnlyEnd extends BaseHook {
      override onPredictEnd(): void {
        log.push("end");
      }
    }
    const agent = makeAgent();
    agent.addHook(new OnlyEnd());
    await agent.predict("s0", QUESTIONS);
    expect(log).toEqual(["end"]);

    const agent2 = makeAgent();
    agent2.addHook(new BaseHook()); // every method is a no-op
    await expect(agent2.predict("s0", QUESTIONS)).resolves.toBeTruthy();
  });

  it("defaults/run before installed and per-call hooks", async () => {
    const log: string[] = [];
    setDefaultHooks(undefined, undefined, () => void log.push("default"));
    try {
      const agent = makeAgent();
      agent.addHook({ onPredictEnd: () => void log.push("installed") });
      await agent.predict("s0", QUESTIONS, { onPredictEnd: () => void log.push("percall") });
      expect(log).toEqual(["default", "installed", "percall"]);
    } finally {
      clearDefaultHooks();
    }
    expect(defaultHooks()).toEqual([]);
  });

  it("defaults/addDefaultHook appends in order; setDefaultHooks replaces", async () => {
    const log: string[] = [];
    addDefaultHook({ onPredictEnd: () => void log.push("a") });
    addDefaultHook({ onPredictEnd: () => void log.push("b") });
    try {
      expect(defaultHooks()).toHaveLength(2);
      await makeAgent().predict("s0", QUESTIONS);
      expect(log).toEqual(["a", "b"]);
    } finally {
      clearDefaultHooks();
    }
    expect(defaultHooks()).toEqual([]);

    // setDefaultHooks replaces rather than appends
    setDefaultHooks({ onPredictEnd: () => void log.push("c") });
    setDefaultHooks({ onPredictEnd: () => void log.push("d") });
    try {
      await makeAgent().predict("s0", QUESTIONS);
      expect(log).toEqual(["a", "b", "d"]);
    } finally {
      clearDefaultHooks();
    }
  });

  it("defaults/read at call time, so hooks set after construction still apply", async () => {
    const log: string[] = [];
    const agent = makeAgent(); // constructed before the default is set
    setDefaultHooks({ onPredictStart: () => void log.push("late") });
    try {
      await agent.predict("s0", QUESTIONS);
      expect(log).toEqual(["late"]);
    } finally {
      clearDefaultHooks();
    }
  });

  it("defaults/cover the router lifecycle (onLoad/onEvict)", async () => {
    const events: [string, string | undefined][] = [];
    class LifeDefaults extends BaseHook {
      override onLoad(ctx: PredictContext): void {
        events.push(["load", ctx.model]);
      }
      override onEvict(ctx: PredictContext): void {
        events.push(["evict", ctx.model]);
      }
    }
    setDefaultHooks([new LifeDefaults()]);
    try {
      const { router } = makeRouter({ maxLoaded: 1 });
      await router.load("english");
      await router.load("multilingual"); // evicts english
      expect(events).toEqual([
        ["load", "english"],
        ["evict", "english"],
        ["load", "multilingual"],
      ]);
    } finally {
      clearDefaultHooks();
    }
  });
});
