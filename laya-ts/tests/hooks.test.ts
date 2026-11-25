import { readFileSync } from "node:fs";
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
  validateHooksTimeout,
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
  const router = new Router({ default: "english", loader, ...extra } as never);
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

  it("awaits an async end hook, so it can rewrite the results after an await", async () => {
    const agent = makeAgent([], {
      async onPredictEnd(ctx) {
        await new Promise((resolve) => setTimeout(resolve, 5));
        for (const r of ctx.results ?? []) r["stamped"] = true;
      },
    });
    const r = (await agent.systemOne("hi", QUESTIONS)) as Record<string, unknown>;
    expect(r["stamped"]).toBe(true);
  });

  it("an async start hook may skip inference after an await", async () => {
    const seen: string[] = [];
    const cached = { model: "cache", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    const agent = makeAgent(seen, {
      async onPredictStart(ctx) {
        await new Promise((resolve) => setTimeout(resolve, 5));
        ctx.skip([cached]);
      },
    });
    expect(await agent.systemOne("hi", QUESTIONS)).toEqual(cached);
    expect(seen).toEqual([]);
  });

  it("waits for each async hook before the next one runs", async () => {
    const order: string[] = [];
    const agent = makeAgent([], {
      hooks: [{
        async onPredictStart() {
          order.push("first:begin");
          await new Promise((resolve) => setTimeout(resolve, 5));
          order.push("first:end");
        },
      }],
    });
    await agent.systemOne("hi", QUESTIONS, { onPredictStart: () => void order.push("second") });
    expect(order).toEqual(["first:begin", "first:end", "second"]);
  });

  it("a rejecting async hook follows hooksRaise: raised by default, a warning when false", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const failing = { async onPredictEnd() { await Promise.resolve(); throw new Error("metrics backend down"); } };
      await expect(makeAgent([], { hooks: [failing] }).systemOne("hi", QUESTIONS)).rejects.toThrow("metrics backend down");
      const r = await makeAgent([], { hooks: [failing], hooksRaise: false }).systemOne("hi", QUESTIONS);
      expect(r.answers).toHaveProperty("q");
      expect(warn).toHaveBeenCalledWith(expect.stringContaining("metrics backend down"));
      await new Promise((resolve) => setTimeout(resolve, 10)); // an unhandled rejection would surface here
    } finally {
      warn.mockRestore();
    }
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

  it("awaits async Router hooks, and a rejecting onRoute hook does not become an unhandled rejection", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const { router } = makeRouter({
        hooksRaise: false,
        hooks: [{
          async onPredictEnd(ctx: PredictContext) {
            await new Promise((resolve) => setTimeout(resolve, 5));
            for (const r of ctx.results ?? []) r["stamped"] = true;
          },
          async onRoute() { await Promise.resolve(); throw new Error("route hook down"); },
        }],
      });
      const r = (await router.predict("plain english text", QUESTIONS)) as unknown as Record<string, unknown>;
      expect(r["stamped"]).toBe(true);
      expect(warn).toHaveBeenCalledWith(expect.stringContaining("route hook down"));
      await new Promise((resolve) => setTimeout(resolve, 10)); // an unhandled rejection would surface here
    } finally {
      warn.mockRestore();
    }
  });

  it("an async onLoad hook may call router.load for the model being loaded without deadlocking", async () => {
    const seen: string[] = [];
    const { router } = makeRouter({
      hooks: [{
        async onLoad(ctx: PredictContext) {
          seen.push("begin");
          await router.load(ctx.model as string);
          seen.push("end");
        },
      }],
    });
    const timeout = new Promise((resolve) => setTimeout(() => resolve("deadlock"), 500));
    expect(await Promise.race([router.load("english"), timeout])).not.toBe("deadlock");
    expect(seen).toEqual(["begin", "end"]);
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

  it("defaults/fire once per Router.predict, not again inside the Agent", async () => {
    const events: string[] = [];
    const tag = (ctx: PredictContext) => (ctx.router ? "router" : "agent");
    setDefaultHooks({
      onPredictStart: (ctx: PredictContext) => void events.push(`start:${tag(ctx)}`),
      onPredictEnd: (ctx: PredictContext) => void events.push(`end:${tag(ctx)}`),
    });
    try {
      const router = new Router();
      router.attach("english", makeAgent() as never);
      await router.predict("s0", QUESTIONS, { model: "english" });
      expect(events).toEqual(["start:router", "end:router"]);
    } finally {
      clearDefaultHooks();
    }
  });

  it("defaults/still fire for a direct Agent call made while a Router request is in flight", async () => {
    const events: string[] = [];
    setDefaultHooks({ onPredictEnd: (ctx: PredictContext) => void events.push(ctx.router ? "router" : "agent") });
    try {
      const router = new Router();
      router.attach("english", makeAgent() as never);
      await Promise.all([
        router.predict("s0", QUESTIONS, { model: "english" }),
        makeAgent().predict("s1", QUESTIONS),
      ]);
      expect(events.filter((e) => e === "router")).toHaveLength(1);
      expect(events.filter((e) => e === "agent")).toHaveLength(1);
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

describe("hook timeout", () => {
  class SlowHook {
    async onPredictStart(): Promise<void> {
      await new Promise((resolve) => setTimeout(resolve, 10_000));
    }
  }

  /** Settle `pending` without an unhandled rejection if the call fails. */
  function capture(pending: Promise<unknown>): Promise<unknown> {
    return pending.then(
      (value) => value,
      (err: unknown) => err,
    );
  }

  it("names the hook and the event when a hook times out", async () => {
    vi.useFakeTimers();
    try {
      const pending = makeAgent([], { hooks: [new SlowHook()], hooksTimeout: 0.05 }).systemOne("hi", QUESTIONS);
      const settled = capture(pending);
      await vi.advanceTimersByTimeAsync(50);
      const err = await settled;
      expect(err).toBeInstanceOf(Error);
      expect((err as Error).name).toBe("TimeoutError");
      expect((err as Error).message).toBe("laya: hook SlowHook.onPredictStart exceeded 0.05s");
    } finally {
      vi.useRealTimers();
    }
  });

  it("clears the timer when an async hook finishes before the deadline", async () => {
    vi.useFakeTimers();
    const armed = vi.spyOn(globalThis, "setTimeout");
    const cleared = vi.spyOn(globalThis, "clearTimeout");
    try {
      class FastHook {
        async onPredictStart(): Promise<void> {
          await Promise.resolve();
        }
      }
      const agent = makeAgent([], { hooks: [new FastHook()], hooksTimeout: 1 });
      await agent.systemOne("hi", QUESTIONS);
      expect(armed).toHaveBeenCalled();
      for (const id of armed.mock.results.map((result) => result.value)) {
        expect(cleared).toHaveBeenCalledWith(id);
      }
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      armed.mockRestore();
      cleared.mockRestore();
      vi.useRealTimers();
    }
  });

  it("does not start a timer for a synchronous hook", async () => {
    vi.useFakeTimers();
    const armed = vi.spyOn(globalThis, "setTimeout");
    try {
      class SyncHook {
        onPredictStart(): void {}
      }
      const agent = makeAgent([], { hooks: [new SyncHook()], hooksTimeout: 0.05 });
      await agent.systemOne("hi", QUESTIONS);
      expect(armed).not.toHaveBeenCalled();
    } finally {
      armed.mockRestore();
      vi.useRealTimers();
    }
  });

  it("hooksRaise false reports the timeout and continues", async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const pending = makeAgent([], {
        hooks: [new SlowHook()],
        hooksTimeout: 0.05,
        hooksRaise: false,
      }).systemOne("hi", QUESTIONS);
      await vi.advanceTimersByTimeAsync(50);
      const r = await pending;
      expect(r.answers).toHaveProperty("q");
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining("laya: hook SlowHook.onPredictStart exceeded 0.05s"),
      );
    } finally {
      warn.mockRestore();
      vi.useRealTimers();
    }
  });

  it("a per-call hooksTimeout overrides the agent default", async () => {
    vi.useFakeTimers();
    try {
      class BoundedHook {
        async onPredictStart(): Promise<void> {
          await new Promise((resolve) => setTimeout(resolve, 200));
        }
      }

      const shortCall = capture(
        makeAgent([], { hooks: [new BoundedHook()], hooksTimeout: 5 }).systemOne("hi", QUESTIONS, {
          hooksTimeout: 0.05,
        }),
      );
      await vi.advanceTimersByTimeAsync(50);
      const err = await shortCall;
      expect(err).toBeInstanceOf(Error);
      expect((err as Error).name).toBe("TimeoutError");
      expect((err as Error).message).toBe("laya: hook BoundedHook.onPredictStart exceeded 0.05s");

      const longCall = makeAgent([], { hooks: [new BoundedHook()], hooksTimeout: 0.05 }).systemOne(
        "hi",
        QUESTIONS,
        { hooksTimeout: 5 },
      );
      await vi.advanceTimersByTimeAsync(200);
      const r = await longCall;
      expect(r.answers).toHaveProperty("q");
    } finally {
      vi.useRealTimers();
    }
  });

  // `laya-java/fixtures/hooks.json` records `laya/hooks.py::validate_timeout` answering every
  // value in `HOOK_TIMEOUT_VALUES` (`gen_fixtures.py:4731`, recorded at :5021-5038): the accepted
  // seconds or the `ValueError` text. `laya-java` replays those rows in `HooksTest.timeoutsMatch()`;
  // until now the TypeScript port had no test that consulted the file, and no gate either --
  // `typeof value === "number" ? value * 1000 : null` installed the deadline for a number and
  // treated everything else as "no limit", so the reference's refusals arrived here as either an
  // instant `TimeoutError` (0, -1, NaN, Infinity) or a missing deadline (a bool, a string, a list).
  //
  // `input` is the Python `repr` of the value, and is the key this test maps on purpose: a value
  // added to the reference's list must land here deliberately rather than fall through untested.
  const TIMEOUT_ROWS = JSON.parse(
    readFileSync(new URL("../../laya-java/fixtures/hooks.json", import.meta.url), "utf8"),
  ).timeouts as {
    input: string;
    accepted: boolean;
    value: number | null;
    seconds: string | null;
    error: string | null;
  }[];

  const TIMEOUT_INPUTS: Record<string, unknown> = {
    None: null,
    "1": 1,
    "2": 2,
    "10": 10,
    "60": 60,
    "1.5": 1.5,
    "0.05": 0.05,
    "1e-07": 1e-7,
    "'0.5'": "0.5",
    "0": 0,
    "0.0": 0,
    "-1": -1,
    "-0.5": -0.5,
    nan: NaN,
    inf: Infinity,
    "-inf": -Infinity,
  };

  /** JavaScript has one number where the reference recorded a Python `int` and a Python `float`.
   *  Both rows still assert the refusal; only the `repr` in its text cannot be reproduced. */
  const REPR_AMBIGUOUS = new Set(["0", "0.0"]);

  /** The message a call refuses with, or a test failure if it did not refuse. `toThrowError` with
   *  a string matches a substring, and the whole point of these rows is the exact text. */
  function refusalOf(build: () => unknown): string {
    try {
      build();
    } catch (err) {
      return (err as Error).message;
    }
    throw new Error("expected a refusal");
  }

  for (const row of TIMEOUT_ROWS) {
    if (!(row.input in TIMEOUT_INPUTS)) {
      it(`${row.input} (UNRECORDED)`, () => {
        throw new Error(
          `hooks.json records the timeout value ${row.input}, which this test maps to no JavaScript ` +
            "value. Add it to TIMEOUT_INPUTS -- leaving it out stops the port being tested for it.",
        );
      });
      continue;
    }
    const value = TIMEOUT_INPUTS[row.input];
    if (row.accepted) {
      it(`${row.input} is accepted as ${row.value}`, () => {
        expect(validateHooksTimeout(value)).toBe(row.value);
      });
      continue;
    }
    it(`${row.input} is refused`, () => {
      expect(refusalOf(() => validateHooksTimeout(value))).toBe(
        REPR_AMBIGUOUS.has(row.input) ? row.error!.replace(/got .*$/, "got 0") : row.error!,
      );
    });
  }

  // Shapes the reference's list does not carry, measured from `validate_timeout` directly.
  it("refuses the shapes a JSON body can hold, naming them as the reference does", () => {
    const PREFIX = "hooks_timeout must be a positive finite number or None";
    expect(refusalOf(() => validateHooksTimeout(true))).toBe(`${PREFIX}, not a bool; got True`);
    expect(refusalOf(() => validateHooksTimeout(false))).toBe(`${PREFIX}, not a bool; got False`);
    expect(refusalOf(() => validateHooksTimeout([1]))).toBe(`${PREFIX}; got [1]`);
    expect(refusalOf(() => validateHooksTimeout({ a: 1 }))).toBe(`${PREFIX}; got {'a': 1}`);
    // A string `float()` cannot read leaks its own message, which the reference does not wrap.
    expect(refusalOf(() => validateHooksTimeout(" "))).toBe("could not convert string to float: ' '");
    expect(refusalOf(() => validateHooksTimeout("abc"))).toBe(
      "could not convert string to float: 'abc'",
    );
    // A spelling `float()` does read is then refused for what it reads, and names the string.
    expect(refusalOf(() => validateHooksTimeout("inf"))).toBe(`${PREFIX}; got 'inf'`);
    expect(refusalOf(() => validateHooksTimeout("0"))).toBe(`${PREFIX}; got '0'`);
    expect(validateHooksTimeout("1e-7")).toBe(1e-7);
  });

  it("refuses a bad timeout when the agent is built, not at the first hook", () => {
    expect(refusalOf(() => makeAgent([], { hooksTimeout: 0 }))).toBe(
      "hooks_timeout must be a positive finite number or None; got 0",
    );
    expect(refusalOf(() => makeAgent([], { hooksTimeout: true }))).toBe(
      "hooks_timeout must be a positive finite number or None, not a bool; got True",
    );
  });

  it("refuses a bad per-call override instead of firing an instant deadline", async () => {
    class NeverHook {
      async onPredictStart(): Promise<void> {
        return new Promise(() => {});
      }
    }
    vi.useFakeTimers();
    try {
      const settled = capture(
        makeAgent([], { hooks: [new NeverHook()] }).systemOne("hi", QUESTIONS, { hooksTimeout: 0 }),
      );
      await vi.advanceTimersByTimeAsync(50);
      const err = await settled;
      expect((err as Error).message).toBe(
        "hooks_timeout must be a positive finite number or None; got 0",
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("reads a string timeout the reference reads and sets that deadline", async () => {
    vi.useFakeTimers();
    try {
      const settled = capture(
        makeAgent([], { hooks: [new SlowHook()], hooksTimeout: "0.5" }).systemOne("hi", QUESTIONS),
      );
      let done = false;
      settled.then(() => {
        done = true;
      });
      await vi.advanceTimersByTimeAsync(400);
      // Half a second has not passed, and before this gate a string timeout never armed a timer
      // at all, so nothing would have fired at any point.
      expect(done).toBe(false);
      await vi.advanceTimersByTimeAsync(100);
      const err = await settled;
      expect((err as Error).name).toBe("TimeoutError");
      expect((err as Error).message).toBe("laya: hook SlowHook.onPredictStart exceeded 0.5s");
    } finally {
      vi.useRealTimers();
    }
  });
});
