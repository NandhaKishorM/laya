import { afterEach, describe, expect, it, vi } from "vitest";
import { Router } from "../src/router.js";
import { clearDefaultHooks, defaultsAlreadyRan, setDefaultHooks } from "../src/hooks.js";

const questions = { q: { type: "noul", instructions: "Is this urgent?" } };
const result = (windows = 3) => ({ model: "test", answers: {}, usage: { input_tokens: 12, output_tokens: 0, windows } });
afterEach(() => { clearDefaultHooks(); vi.restoreAllMocks(); });

describe("Router.predictLong", () => {
  it("routes and loads once, forwards language and scan options, wraps the document", async () => {
    const calls: string[] = [];
    setDefaultHooks({ onPredictStart: () => { calls.push("default"); } });
    const predictLong = vi.fn(async (_s, _q, opts) => {
      expect(defaultsAlreadyRan(opts)).toBe(true);
      calls.push("scan");
      return result();
    });
    const loader = vi.fn(async () => ({ predictLong }));
    const router = new Router({ loader, hooks: {
      onRoute: () => { calls.push("route"); },
      onPredictStart: (ctx) => { expect(ctx.states).toHaveLength(1); calls.push("start"); },
      onPredictEnd: () => { calls.push("end"); },
    } });
    const out = await router.predictLong("Please refund the duplicate charge on my invoice", questions,
      { window: 80, stride: 40, batchSize: 2, aggregate: "auto" });
    expect(loader).toHaveBeenCalledTimes(1);
    expect(predictLong).toHaveBeenCalledTimes(1);
    expect(predictLong.mock.calls[0][2]).toMatchObject({ lang: "en", window: 80, stride: 40, batchSize: 2, aggregate: "auto" });
    expect(out.routing.model).toBe("english");
    expect(out.usage.windows).toBe(3);
    expect(calls).toEqual(["route", "default", "start", "scan", "end"]);
  });

  it("uses the post-start agent, state, questions and detected language", async () => {
    const original = { predictLong: vi.fn() };
    const changed = { predictLong: vi.fn(async () => result(2)) };
    const router = new Router({ loader: async () => original });
    const changedQuestions = { other: { type: "noul", instructions: "Changed?" } };
    await router.predictLong("hello", questions, { onPredictStart(ctx) {
      ctx.agent = changed;
      ctx.states = ["changed"];
      ctx.questions = changedQuestions;
      ctx.decision = { ...ctx.decision, detection: { language: "de" } };
      ctx.maxLen = 1; ctx.headMaxLen = 1;
    } });
    expect(original.predictLong).not.toHaveBeenCalled();
    expect(changed.predictLong).toHaveBeenCalledWith("changed", changedQuestions, expect.objectContaining({ lang: "de" }));
    expect(changed.predictLong.mock.calls[0]).toHaveLength(3);
    const opts = changed.predictLong.mock.calls[0][2] as Record<string, unknown>;
    expect(opts).not.toHaveProperty("maxLen");
    expect(opts).not.toHaveProperty("headMaxLen");
  });

  it("copies a skipped cached document and usage, warns and reports zero windows", async () => {
    const cached = { model: "cached", answers: {}, usage: { input_tokens: 8, output_tokens: 0 } };
    const predictLong = vi.fn();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const router = new Router({ loader: async () => ({ predictLong }) });
    const out = await router.predictLong("hello", questions, { onPredictStart: (ctx) => ctx.skip([cached]) });
    expect(predictLong).not.toHaveBeenCalled();
    expect(out).not.toBe(cached);
    expect(out.usage).not.toBe(cached.usage);
    expect(out.usage.windows).toBe(0);
    expect(cached.usage).not.toHaveProperty("windows");
    expect(cached).not.toHaveProperty("routing");
    expect(warn).toHaveBeenCalled();
  });

  it.each([{ results: [] }, { results: [result(), result()] }])("rejects ambiguous hook document results", async ({ results }) => {
    const router = new Router({ loader: async () => ({}) });
    await expect(router.predictLong("hello", questions, { hooksRaise: false,
      onPredictStart: (ctx) => ctx.skip(results),
    })).rejects.toThrow(/exactly one/);
  });

  it("propagates missing scan support and inference errors even when hooks are suppressed", async () => {
    const systemOne = vi.fn(async () => result());
    const router = new Router({ loader: async () => ({ systemOne }) });
    await expect(router.predictLong("hello", questions, { hooksRaise: false })).rejects.toThrow(/implement predictLong/);
    expect(systemOne).not.toHaveBeenCalled();
    router.attach("english", { predictLong: async () => { throw new Error("scan failed"); } });
    await expect(router.predictLong("hello", questions, { model: "english", hooksRaise: false })).rejects.toThrow("scan failed");
  });

  it("does not apply hook timeouts to inference", async () => {
    const router = new Router({ loader: async () => ({ predictLong: async () => {
      await new Promise((resolve) => setTimeout(resolve, 30));
      return result();
    } }) });
    await expect(router.predictLong("hello", questions, { hooksTimeout: 0.001, hooksRaise: true })).resolves.toMatchObject({ usage: { windows: 3 } });
  });

  it("keeps concurrent ordinary and long calls isolated, including reentrant ordinary prediction", async () => {
    const calls: string[] = [];
    const agent = {
      systemOne: vi.fn(async (state) => { calls.push(`ordinary:${state}`); return result(1); }),
      predictLong: vi.fn(async (state) => { calls.push(`long:${state}`); return result(4); }),
    };
    const router = new Router({ loader: async () => agent });
    const [long, ordinary] = await Promise.all([
      router.predictLong("long", questions, { onPredictStart: async () => { await router.predict("nested", questions); } }),
      router.predict("ordinary", questions),
    ]);
    expect(long.usage.windows).toBe(4);
    expect(ordinary.model).toBe("test");
    expect(calls.sort()).toEqual(["long:long", "ordinary:nested", "ordinary:ordinary"]);
  });

  it("validates the final document supplied by an end hook", async () => {
    const router = new Router({ loader: async () => ({ predictLong: async () => result() }) });
    await expect(router.predictLong("hello", questions, { hooksRaise: false, onPredictEnd(ctx) {
      ctx.results = [{ ...result(), usage: { input_tokens: 0, output_tokens: 0, windows: -1 } }];
    } })).rejects.toThrow(/windows/);
  });

  it("refuses an end-hook replacement that drops the promised routing decision", async () => {
    const router = new Router({ loader: async () => ({ predictLong: async () => result() }) });
    await expect(router.predictLong("hello", questions, { hooksRaise: false, onPredictEnd(ctx) {
      ctx.results = [result()];
    } })).rejects.toThrow(/routing decision/);
  });

  it("validates detection metadata in a hook-supplied routing decision", async () => {
    const router = new Router({ loader: async () => ({ predictLong: async () => result() }) });
    await expect(router.predictLong("hello", questions, { onPredictEnd(ctx) {
      (ctx.results![0].routing as Record<string, unknown>).detection = {};
    } })).rejects.toThrow(/routing decision/);
  });

  it.each([{ window: 0 }, { stride: 1.5 }, { batchSize: -1 }, { window: Number.MAX_SAFE_INTEGER + 1 }])("rejects invalid scan options before loading: %j", async (opts) => {
    const loader = vi.fn();
    await expect(new Router({ loader }).predictLong("hello", questions, opts)).rejects.toThrow();
    expect(loader).not.toHaveBeenCalled();
  });
});
