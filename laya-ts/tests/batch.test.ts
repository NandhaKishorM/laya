import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import { Router } from "../src/router.js";

function fakeProvider(counter: { enc: number }) {
  return {
    async runEncoder(batch: any) {
      counter.enc++;
      return { lastHidden: batch.inputIds.map((row: number[]) => row.map(() => [1, 0])) };
    },
    async runHead(_h: unknown, batch: any) {
      return {
        logits: batch.markerPos.map((m: number[]) => m.map((_, i) => 2 - i * 0.5)),
        act: batch.markerPos.map(() => [3, 0]),
      };
    },
  };
}

const Q: any = {
  d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } },
};

describe("predictMany", () => {
  it("batches N states into one encoder run with per-state results", async () => {
    const counter = { enc: 0 };
    const a = new Agent({ provider: fakeProvider(counter) } as any);
    const out = await (a as any).predictMany(["hi one", "hi two"], Q);
    expect(out).toHaveLength(2);
    expect(out[0].answers.d.choice).toBe("x");
    expect(out[1].answers.d.choice).toBe("x");
    expect(counter.enc).toBe(1);
  });
  it("matches looped systemOne", async () => {
    const c1 = { enc: 0 };
    const c2 = { enc: 0 };
    const a1 = new Agent({ provider: fakeProvider(c1) } as any);
    const a2 = new Agent({ provider: fakeProvider(c2) } as any);
    const states = ["alpha", "beta", "gamma"];
    const batched = await (a1 as any).predictMany(states, Q);
    const looped = [];
    for (const s of states) looped.push(await a2.systemOne(s, Q));
    expect(batched.map((r: any) => r.answers)).toEqual(looped.map((r) => r.answers));
  });
  it("empty states and empty questions", async () => {
    const a = new Agent({ provider: fakeProvider({ enc: 0 }) } as any);
    expect(await (a as any).predictMany([], Q)).toEqual([]);
    const out = await (a as any).predictMany(["hi"], {});
    expect(out[0].answers).toEqual({});
  });
});

describe("router predictMany", () => {
  function loaderWithCounts(calls: Record<string, number>) {
    return async (name: string) => {
      calls[name] = (calls[name] ?? 0) + 1;
      const agent = new Agent({ provider: fakeProvider({ enc: 0 }) } as any);
      return agent;
    };
  }
  it("routes each state, loads once per model, keeps order + routing keys", async () => {
    const calls: Record<string, number> = {};
    const r = new Router({ loader: loaderWithCounts(calls) } as any);
    const states = [
      "Please refund the duplicate charge",
      "मुझसे दो बार शुल्क लिया गया",
      "Please refund again",
    ];
    const out = await (r as any).predictMany(states, Q);
    expect(out).toHaveLength(3);
    expect(out[0].routing.model).toBe("english");
    expect(out[1].routing.model).toBe("multilingual");
    expect(out[2].routing.model).toBe("english");
    expect(calls["english"]).toBe(1);
    expect(calls["multilingual"]).toBe(1);
  });
  it("matches looped predict answers", async () => {
    const r1 = new Router({ loader: loaderWithCounts({}) } as any);
    const r2 = new Router({ loader: loaderWithCounts({}) } as any);
    const states = ["hello there", "Der Kunde wurde zweimal belastet"];
    const batched = await (r1 as any).predictMany(states, Q);
    const looped = [];
    for (const s of states) looped.push(await r2.predict(s, Q));
    expect(batched.map((x: any) => x.answers)).toEqual(looped.map((x) => x.answers));
    expect(batched.map((x: any) => x.routing.model)).toEqual(looped.map((x) => x.routing.model));
  });
});
