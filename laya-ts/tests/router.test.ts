import { describe, expect, it, vi } from "vitest";
import { Router, normaliseName } from "../src/router.js";
describe("router", () => {
  it("normalises aliases", () => { expect(normaliseName("en")).toBe("english"); });
  it("rejects unknown model", () => { expect(() => normaliseName("jev-1")).toThrow(); });
  it.each(["refund", "12345 ???", { body: "refund" }])("uses multilingual for unidentified state %j", (state) => {
    const r = new Router();
    const decision = r.route(state, {});
    expect(r.default).toBe("multilingual");
    expect(decision.detection?.languageUndecided).toBe(true);
    expect(decision.model).toBe("multilingual");
    expect(decision.reason).toContain("using default (multilingual)");
    expect(r.loaded).toEqual([]);
  });
  it.each(["english", "multilingual"])("applies default %s only to unidentified text", (fallback) => {
    const r = new Router({ default: fallback });
    expect(r.route("refund", {}).model).toBe(fallback);
    expect(r.route("Please refund the duplicate charge on the invoice", {}).model).toBe("english");
    expect(r.route("मुझसे दो बार शुल्क लिया गया", {}).model).toBe("multilingual");
    expect(r.route("refund", {}, { lang: "en" }).model).toBe("english");
    expect(r.route("refund", {}, { model: "typed-decisions" }).model).toBe("typed-decisions");
  });
  it("uses the default in a mixed routing batch", () => {
    const r = new Router();
    const decisions = r.routeBatch([
      { state: "refund", questions: {} },
      { state: "Please refund the duplicate charge on the invoice", questions: {} },
    ]);
    expect(decisions.map((d) => d.model)).toEqual(["multilingual", "english"]);
    expect(r.loaded).toEqual([]);
  });
  it("loads the multilingual checkpoint when predicting unidentified text", async () => {
    const loader = vi.fn(async () => ({
      async systemOne() {
        return { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 5, output_tokens: 0 } };
      },
    }));
    const r = new Router({ loader });
    const result = await r.predict("refund", { q: { type: "noul", instructions: "?" } });
    expect(loader).toHaveBeenCalledWith("multilingual", r.models.multilingual);
    expect(result.routing.model).toBe("multilingual");
    expect(r.loaded).toEqual(["multilingual"]);
  });
  it("routes hindi script to multilingual without loading", () => {
    const r = new Router();
    expect(r.route({ body: "मुझसे दो बार शुल्क लिया गया" }, {}).model).toBe("multilingual");
    expect(r.loaded).toEqual([]);
  });
  it.each([
    ["Please send me the café menu today please", "english"],
    ["Send the invoice to José before Friday", "english"],
    ["Grüße aus Köln, wir melden uns wegen der Rechnung", "multilingual"],
    ["sluk lyset i soveværelset", "multilingual"],
  ])("routes %s to %s", (text, model) => {
    expect(new Router().route(text, {}).model).toBe(model);
  });
  it("explicit model wins", () => {
    expect(new Router().route("hi", {}, { model: "typed-decisions" }).model).toBe("typed-decisions");
  });
  it("lang_guess callable routes", () => {
    const r = new Router();
    expect(r.route("text", {}, { langGuess: () => "ro" }).model).toBe("multilingual");
  });
  it("shares concurrent loads and emits lifecycle hooks once per cached model", async () => {
    const events: string[] = [];
    const loader = vi.fn(async (name: string) => ({ name }));
    const r = new Router({
      maxLoaded: 1,
      loader,
      hooks: [{
        onLoad: (ctx) => events.push(`load:${ctx.model}`),
        onEvict: (ctx) => events.push(`evict:${ctx.model}`),
      }],
    });

    const [first, second] = await Promise.all([r.load("en"), r.load("english")]);
    expect(first).toBe(second);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(r.loaded).toEqual(["english"]);
    expect(r._agents.size).toBe(1);
    await r.load("multilingual");
    expect(events).toEqual(["load:english", "evict:english", "load:multilingual"]);
    expect(r.loaded).toEqual(["multilingual"]);
  });
  function buildRouter(options: Record<string, unknown>): Router {
    return new Router(options as never);
  }
  // The residency cap is the one Router option whose misread fails silently: `Number()` gives NaN
  // for 'abc', `{}` or NaN itself, `Math.max(1, NaN)` is NaN, and `_evict`'s
  // `while (this._order.length > this.maxLoaded)` is then forever false -- every checkpoint a
  // process loads stays resident, which is the exact failure `maxLoaded` exists to prevent. The
  // reference cannot reach that state: `laya/router.py`'s `Router.__init__` is
  // `max(1, int(max_loaded))` and `int()` returns an integer or raises. Both tables are the
  // oracle's, measured by constructing `laya.router.Router` with every value listed here.
  const MAX_LOADED_READS: [string, unknown, number][] = [
    ["'2'", "2", 2],
    ["' 2 '", " 2 ", 2],
    ["'1_0'", "1_0", 10],
    ["'٢'", "٢", 2],
    ["'２'", "２", 2],
    ["'١٢'", "١٢", 12],
    ["'١٢٣'", "١٢٣", 123],
    ["'-٢'", "-٢", 1],
    ["2.7", 2.7, 2],
    ["1e300", 1e300, 1e300],
    ["10n", 10n, 10],
    ["0", 0, 1],
    ["-3", -3, 1],
    ["True", true, 1],
    ["False", false, 1],
    ["undefined", undefined, 2],
    ["null", null, 2],
  ];
  const MAX_LOADED_REFUSALS: [string, unknown, string][] = [
    ["'abc'", "abc", "'abc'"],
    ["'2.5'", "2.5", "'2.5'"],
    ["'1e3'", "1e3", "'1e3'"],
    ["'0x10'", "0x10", "'0x10'"],
    ["''", "", "''"],
    ["'  '", "  ", "'  '"],
    ["'_1'", "_1", "'_1'"],
    ["'1_'", "1_", "'1_'"],
    ["'1__0'", "1__0", "'1__0'"],
    ["NaN", NaN, "nan"],
    ["Infinity", Infinity, "inf"],
    ["-Infinity", -Infinity, "-inf"],
    ["1e400", 1e400, "inf"],
    ["{}", {}, "{}"],
    ["[]", [], "[]"],
    ["[3]", [3], "[3]"],
  ];
  /** The message the constructor raises, or a failure that it raised nothing at all. */
  function maxLoadedRefusal(options: Record<string, unknown>): string {
    try {
      buildRouter(options);
    } catch (err) {
      return (err as Error).message;
    }
    throw new Error("expected the Router constructor to refuse this max_loaded");
  }
  for (const [label, value, expected] of MAX_LOADED_READS) {
    it(`reads maxLoaded ${label} as the reference does`, () => {
      expect(buildRouter({ maxLoaded: value }).maxLoaded).toBe(expected);
    });
  }
  for (const [label, value, repr] of MAX_LOADED_REFUSALS) {
    it(`refuses maxLoaded ${label} instead of disabling the cap`, () => {
      const message = `max_loaded must be a count of checkpoints to keep resident, got ${repr}`;
      expect(maxLoadedRefusal({ maxLoaded: value })).toBe(message);
      expect(maxLoadedRefusal({ max_loaded: value })).toBe(message);
    });
  }
  it("evicts down to a cap it read from text", async () => {
    const loads: string[] = [];
    const r = buildRouter({
      maxLoaded: "2",
      loader: (name: string) => {
        loads.push(name);
        return { name };
      },
    });
    expect(r.maxLoaded).toBe(2);
    for (const name of ["en", "ml", "typed"]) await r.load(name);
    expect(loads).toEqual(["english", "multilingual", "typed-decisions"]);
    expect(r.loaded).toEqual(["multilingual", "typed-decisions"]);
  });
  it("raises a read cap to fit attached checkpoints", () => {
    const r = buildRouter({ maxLoaded: "1" });
    r.attach("en", { name: "english" });
    expect(r.maxLoaded).toBe(1);
    r.attach("ml", { name: "multilingual" });
    expect(r.maxLoaded).toBe(2);
  });
  it("clears an unsuccessful in-flight load so the model can be retried", async () => {
    const agent = { name: "english" };
    const loader = vi.fn()
      .mockRejectedValueOnce(new Error("load failed"))
      .mockResolvedValue(agent);
    const r = new Router({ loader });

    const results = await Promise.allSettled([r.load("en"), r.load("english")]);
    expect(results.map((result) => result.status)).toEqual(["rejected", "rejected"]);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(await r.load("english")).toBe(agent);
    expect(loader).toHaveBeenCalledTimes(2);
    expect(r.loaded).toEqual(["english"]);
  });
  it("rejects invalid inputs in predict before loading any checkpoint", async () => {
    const loader = vi.fn();
    const r = new Router({ loader });

    await expect(r.predict(null, { q: { type: "noul", instructions: "?" } })).rejects.toThrow(
      TypeError,
    );
    await expect(r.predict(null, { q: { type: "noul", instructions: "?" } })).rejects.toThrow(
      /state must not be None/,
    );

    await expect(r.predict("text", [] as never)).rejects.toThrow(TypeError);
    await expect(r.predict("text", [] as never)).rejects.toThrow(
      /questions must be a dict of question id -> definition/,
    );

    await expect(r.predict("text", null as never)).rejects.toThrow(TypeError);
    await expect(r.predict("text", null as never)).rejects.toThrow(
      /questions must be a dict of question id -> definition/,
    );

    expect(loader).not.toHaveBeenCalled();
    expect(r.loaded).toEqual([]);
  });
  it("rejects invalid questions type in route", () => {
    const r = new Router();
    expect(() => r.route("text", [] as never)).toThrow(TypeError);
    expect(() => r.route("text", [] as never)).toThrow(
      /questions must be a dict of question id -> definition/,
    );
  });
  it("dispatches onError and onPredictEnd when load fails inside predict lifecycle", async () => {
    const loader = vi.fn(async () => {
      throw new Error("checkpoint fetch failed");
    });
    const r = new Router({ loader });
    const events: string[] = [];

    await expect(
      r.predict("text", { q: { type: "noul", instructions: "?" } }, {
        hooks: [
          {
            onError(ctx) {
              events.push(`error:${(ctx.error as Error).message}`);
            },
            onPredictEnd(ctx) {
              events.push(`end:${(ctx.error as Error).message}`);
            },
          },
        ],
      }),
    ).rejects.toThrow("checkpoint fetch failed");

    expect(events).toEqual(["error:checkpoint fetch failed", "end:checkpoint fetch failed"]);
  });
  it("validates question definitions before calling systemOne", async () => {
    const customAgent = {
      systemOne: vi.fn(async () => ({ model: "custom", answers: {}, usage: {} })),
    };
    const r = new Router({ loader: vi.fn(async () => customAgent) });

    await expect(
      r.predict("text", { q: { type: "bogus", instructions: "?" } } as never),
    ).rejects.toThrow(/unknown type/);

    await expect(
      r.predict("text", { "": { type: "noul", instructions: "?" } } as never),
    ).rejects.toThrow(/question id must be a non-empty string/);

    await expect(
      r.predict("text", { q: { type: "noul", instructions: "" } } as never),
    ).rejects.toThrow(/instructions.*must not be empty/);

    expect(customAgent.systemOne).not.toHaveBeenCalled();
  });
  it("allows onPredictStart hook to complete an incomplete question or skip", async () => {
    const customAgent = {
      systemOne: vi.fn(async () => ({ model: "custom", answers: {}, usage: {} })),
    };
    const r = new Router({ loader: vi.fn(async () => customAgent) });

    const res = await r.predict("text", { q: { type: "noul" } as never }, {
      onPredictStart: (ctx) => {
        ctx.questions = { q: { type: "noul", instructions: "Added by hook?" } };
      },
    });
    expect(res.routing).toBeDefined();
    expect(customAgent.systemOne).toHaveBeenCalled();

    const skipped = await r.predict("text", { q: { type: "noul" } as never }, {
      onPredictStart: (ctx) => {
        ctx.skip([{ model: "cached", answers: {}, usage: {} }]);
      },
    });
    expect(skipped.routing).toBeDefined();
    expect(skipped.model).toBe("cached");
  });

  // Python's `Router` takes `hooks_timeout` (laya/router.py: `self.hooks_timeout = None if
  // hooks_timeout is None else validate_timeout(hooks_timeout)`), passes it to the `on_load` and
  // lifecycle dispatches (`timeout=self.hooks_timeout`), and in `predict` resolves
  // `self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)`. The TS
  // Router had no such field: only a limit the caller named on the long path was read, and a hook
  // that never settled held `load()` or `predict()` open forever.
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
  const noul = { q: { type: "noul", instructions: "?" } };
  const stubAgent = {
    systemOne: async () => ({ model: "stub", answers: { q: "yes" }, usage: {} }),
    predictLong: async () => ({ model: "stub", answers: {}, usage: { input_tokens: 12, output_tokens: 0, windows: 1 } }),
  };

  it("refuses a Router-level hooksTimeout the reference would refuse", () => {
    // `validate_timeout` decides; the Router only forwards to it, on the constructor line.
    class Ok { onPredictStart() {} }
    expect(new Router({ hooksTimeout: 0.5 }).hooksTimeout).toBe(0.5);
    expect(new Router({ hooks_timeout: "0.5" }).hooksTimeout).toBe(0.5);
    expect(new Router({}).hooksTimeout).toBeNull();
    expect(new Router({ hooksTimeout: null }).hooksTimeout).toBeNull();
    expect(new Router({ hooksTimeout: undefined }).hooksTimeout).toBeNull();
    expect(new Router({ hooksTimeout: 2, hooks: [new Ok()] }).hooksTimeout).toBe(2);
    for (const bad of [true, false, NaN, Infinity, -Infinity, 0, -0.5, {}, [], "soon", () => 1]) {
      expect(() => new Router({ hooksTimeout: bad as never })).toThrow();
    }
    // The two messages are the reference's own, and which one fires is the contract: a bool reads
    // as a wrong type, an unreadable string as a failed coercion.
    expect(() => new Router({ hooksTimeout: true as never }))
      .toThrow("hooks_timeout must be a positive finite number or None, not a bool; got True");
    expect(() => new Router({ hooksTimeout: "soon" as never }))
      .toThrow("could not convert string to float: 'soon'");
    // A string the reference can read as float is read, not refused -- `float("0.5")`.
    expect(new Router({ hooksTimeout: "1.5" }).hooksTimeout).toBe(1.5);
  });

  it("bounds an onLoad hook that overruns the Router's limit", async () => {
    class SlowLoad {
      async onLoad() { await sleep(80); }
    }
    const timed = new Router({ hooksTimeout: 0.01, loader: async () => stubAgent, hooks: [new SlowLoad()] });
    await expect(timed.load("english")).rejects.toThrow("laya: hook SlowLoad.onLoad exceeded 0.01s");
    const patient = new Router({ hooksTimeout: 0.5, loader: async () => stubAgent, hooks: [new SlowLoad()] });
    await expect(patient.load("english")).resolves.toBeDefined();
  });

  it("bounds an onEvict hook, which fires from load rather than from predict", async () => {
    class SlowEvict {
      async onEvict() { await sleep(80); }
    }
    const timed = new Router({ hooksTimeout: 0.01, maxLoaded: 1, loader: async () => stubAgent, hooks: [new SlowEvict()] });
    await timed.load("english");
    await expect(timed.load("multilingual")).rejects.toThrow("laya: hook SlowEvict.onEvict exceeded 0.01s");
  });

  it("bounds a per-call start hook with the Router's own limit", async () => {
    const timed = new Router({ hooksTimeout: 0.01, loader: async () => stubAgent });
    await expect(
      timed.predict("text", noul, { onPredictStart: async () => { await sleep(80); } }),
    ).rejects.toThrow(/exceeded 0\.01s/);
    const unlimited = new Router({ loader: async () => stubAgent });
    await expect(
      unlimited.predict("text", noul, { onPredictStart: async () => { await sleep(20); } }),
    ).resolves.toMatchObject({ model: "stub" });
  });

  it("lets a per-call limit replace the Router's, not shorten it", async () => {
    // `self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)`: a caller
    // who names a deadline on the call gets exactly that deadline, looser or tighter.
    const tight = new Router({ hooksTimeout: 0.01, loader: async () => stubAgent });
    await expect(
      tight.predict("text", noul, { hooksTimeout: 0.5, onPredictStart: async () => { await sleep(60); } }),
    ).resolves.toMatchObject({ model: "stub" });
    const loose = new Router({ hooksTimeout: 0.5, loader: async () => stubAgent });
    await expect(
      loose.predict("text", noul, { hooksTimeout: 0.01, onPredictStart: async () => { await sleep(60); } }),
    ).rejects.toThrow(/exceeded 0\.01s/);
    await expect(loose.predict("text", noul, { hooksTimeout: true as never })).rejects.toThrow(/hooks_timeout/);
  });

  it("gives the long path the Router default it used to miss", async () => {
    // Before, `predict` read a limit only from `scan.hooksTimeout` -- so `predictLong` with a hook
    // that never settled ran to its own pace whatever the Router was built with.
    const timed = new Router({ hooksTimeout: 0.01, loader: async () => stubAgent });
    await expect(
      timed.predictLong("text", noul, { onPredictStart: async () => { await sleep(80); } }),
    ).rejects.toThrow(/exceeded 0\.01s/);
    await expect(
      timed.predictLong("text", noul, { hooksTimeout: 0.5, onPredictStart: async () => { await sleep(60); } }),
    ).resolves.toMatchObject({ model: "stub" });
  });

  it("reports a hook timeout instead of throwing when hooksRaise is false", async () => {
    // `raiseErrors: false` keeps the reference's behaviour: the overrun is reported, not raised,
    // and the prediction still completes -- a hook failure is not the request's failure. The
    // warning is the witness that the deadline fired at all, so this row fails at the base too.
    const warned: string[] = [];
    const spy = vi.spyOn(console, "warn").mockImplementation((...args: unknown[]) => {
      warned.push(args.join(" "));
    });
    try {
      const r = new Router({ hooksTimeout: 0.01, loader: async () => stubAgent });
      const result = await r.predict("text", noul, {
        hooksRaise: false,
        onPredictStart: async () => { await sleep(80); },
      });
      expect(result.model).toBe("stub");
      expect(warned.filter((line) => line.includes("exceeded 0.01s")).length, warned.join("\n")).toBe(1);
    } finally {
      spy.mockRestore();
    }
  });
});
