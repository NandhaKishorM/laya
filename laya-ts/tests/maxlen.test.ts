import { describe, expect, it } from "vitest";
import { maxOf } from "../src/common.js";
import { Router } from "../src/router.js";

describe("maxOf", () => {
  it("returns max without spread overflow", () => {
    expect(maxOf([1, 5, 3], 1)).toBe(5);
    expect(maxOf([], 1)).toBe(1);
  });
  it("handles 300k rows without RangeError", () => {
    const lens = new Array(300_000).fill(1);
    lens[123456] = 77;
    expect(maxOf(lens, 1)).toBe(77);
  });
});

// ------------------------------------------------------------------ Router.predict forwarding
//
// `PredictOptions` documents per-call token-budget overrides and `Agent.systemOne` honours them,
// but `Router.predict` never read them: the context was built without them and `agentOpts` carried
// only `lang`/`minConfidence`, so the primary entry point silently used the checkpoint default. A
// start hook's `ctx.maxLen` was dropped the same way. Python seeds `ctx.max_len` and forwards
// `ctx.max_len`/`ctx.head_max_len` when set (`laya/router.py`), which is what these pin.
describe("Router.predict token budgets", () => {
  const Q = { q: { type: "noul", instructions: "?" } } as any;

  function recordingRouter(seen: any[]) {
    const stub = {
      async systemOne(_state: unknown, _q: unknown, opts?: any) {
        seen.push(opts ?? {});
        return { model: "m", answers: {}, usage: { input_tokens: 1, output_tokens: 0 } };
      },
    };
    return new Router({ default: "english", loader: () => stub } as any);
  }

  it("forwards maxLen and headMaxLen to the agent", async () => {
    const seen: any[] = [];
    const r = recordingRouter(seen);
    await r.predict("please refund my ticket", Q, { maxLen: 64, headMaxLen: 100 });
    expect(seen[0].maxLen).toBe(64);
    expect(seen[0].headMaxLen).toBe(100);
  });

  it("leaves both unset when the caller did not ask, so the agent config applies", async () => {
    const seen: any[] = [];
    const r = recordingRouter(seen);
    await r.predict("please refund my ticket", Q);
    expect("maxLen" in seen[0]).toBe(false);
    expect("headMaxLen" in seen[0]).toBe(false);
  });

  it("lets a start hook set the budget, the way it can rewrite state and questions", async () => {
    const seen: any[] = [];
    const r = recordingRouter(seen);
    await r.predict("please refund my ticket", Q, {
      onPredictStart(ctx) {
        ctx.maxLen = 48;
      },
    });
    expect(seen[0].maxLen).toBe(48);
  });

  it("forwards no budget when a start hook leaves the fields alone", async () => {
    // An absent override must not become an explicit one just because a hook ran.
    const seen: any[] = [];
    const r = recordingRouter(seen);
    await r.predict("please refund my ticket", Q, { onPredictStart() {} });
    expect("maxLen" in seen[0]).toBe(false);
  });
});
