import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
const fakeProvider = () => ({
  async runEncoder(b: any) {
    const n = b?.inputIds?.length ?? 1;
    return { lastHidden: Array.from({ length: n }, () => [[1, 0], [0, 1]]) };
  },
  async runHead(_h: any, batch?: any) {
    const n = batch?.inputIds?.length ?? 1;
    return {
      logits: Array.from({ length: n }, () => [2, 0, 1, 0]),
      act: Array.from({ length: n }, () => [3, 0]),
    };
  },
});
describe("agent", () => {
  it("empty questions skip forward pass", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    expect(await a.systemOne("hi", {})).toEqual(
      expect.objectContaining({ answers: {} }));
  });
  it("choice picks argmax with temp + confidence", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const r: any = await a.systemOne("hi", {
      d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } } });
    expect(r.answers.d.choice).toBe("x");
    expect(r.usage.input_tokens).toBeGreaterThan(0);
  });
  it("rejects bad question with qid", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    await expect(a.systemOne("hi", { q: { type: "nope" } as any })).rejects.toThrow('question "q"');
  });
  it("score legend stringifies non-string levels (#302)", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const r: any = await a.systemOne("hi", {
      s: { type: "score", instructions: "scale?", criteria: [1, 2, 3] as any } });
    const ans = r.answers.s;
    expect(ans.legend).toEqual({ "0": "1", "1": "2", "2": "3" });
    for (const val of Object.values(ans.legend)) {
      expect(typeof val).toBe("string");
    }
  });

  it("non-dict questions raise a clear TypeError", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    for (const method of ["predict", "systemOne"] as const) {
      for (const bad of [null, [], "not-a-dict", 123, new Map(), new Set()]) {
        await expect(a[method]("hello", bad as any)).rejects.toThrow(TypeError);
        await expect(a[method]("hello", bad as any)).rejects.toThrow("questions must be a dict");
      }
    }
  });

  it("none state raises TypeError instead of answering the literal null", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const q = { q: { type: "noul", instructions: "Is it true?" } } as any;
    for (const method of ["predict", "systemOne"] as const) {
      for (const bad of [null, undefined]) {
        await expect(a[method](bad, q)).rejects.toThrow(TypeError);
        await expect(a[method](bad, q)).rejects.toThrow("state must not be None");
      }
    }
  });

  it("predictBatch validates state array and rejects non-array", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const q = { q: { type: "noul", instructions: "Is it true?" } } as any;
    await expect(a.predictBatch("not-an-array" as any, q)).rejects.toThrow(TypeError);
    await expect(a.predictBatch("not-an-array" as any, q)).rejects.toThrow(
      "predictBatch expects an array of states",
    );
    await expect(a.predictBatch(["valid", null], q)).rejects.toThrow("state must not be None");
    await expect(a.predictBatch(new Array(2), q)).rejects.toThrow("state must not be None");
    const results = await a.predictBatch(["s1", "s2"], q);
    expect(results).toHaveLength(2);
    expect(results[0].answers.q.type).toBe("noul");
  });

  it("start hook can normalize questions or states before validation", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const result = await a.predict("hello", null as any, {
      onPredictStart: (ctx) => {
        ctx.questions = { q: { type: "noul", instructions: "Is it true?" } };
      },
    });
    expect(result.answers.q.type).toBe("noul");
  });
});

