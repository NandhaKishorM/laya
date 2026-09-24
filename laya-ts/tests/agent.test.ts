import { describe, expect, it } from "vitest";
import { Agent, defaultTokenizer, type ChoiceAnswer, type NoulAnswer } from "../src/agent.js";
const fakeProvider = () => ({
  async runEncoder(_b: any) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: any) { return { logits: [[2, 0]], act: [[3, 0]] }; },
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

  it("empty batch returns [] without calling provider", async () => {
    const log: { encoderBatches: number[][][]; headCalls: number } = { encoderBatches: [], headCalls: 0 };
    const dynamicProvider = {
      async runEncoder(b: { inputIds: number[][] }) {
        log.encoderBatches.push(b.inputIds);
        return { lastHidden: b.inputIds.map((row) => row.map(() => [1, 0])) };
      },
      async runHead(_h: unknown, b: { markerPos: number[][] }) {
        log.headCalls++;
        return {
          logits: b.markerPos.map((m) => m.map((_, i) => i + 1)),
          act: b.markerPos.map(() => [3, 0]),
        };
      },
    };
    const a = new Agent({ provider: dynamicProvider } as any);
    const res = await a.predictBatch([], {
      d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } },
    });
    expect(res).toEqual([]);
    expect(log.encoderBatches.length).toBe(0);
    expect(log.headCalls).toBe(0);
  });

  it("evaluates multiple states in a single shared forward pass", async () => {
    const log: { encoderBatches: number[][][]; headCalls: number } = { encoderBatches: [], headCalls: 0 };
    const dynamicProvider = {
      async runEncoder(b: { inputIds: number[][] }) {
        log.encoderBatches.push(b.inputIds);
        return { lastHidden: b.inputIds.map((row) => row.map(() => [1, 0])) };
      },
      async runHead(_h: unknown, b: { markerPos: number[][] }) {
        log.headCalls++;
        return {
          logits: b.markerPos.map((m) => m.map((_, i) => i + 1)),
          act: b.markerPos.map(() => [3, 0]),
        };
      },
    };
    const a = new Agent({ provider: dynamicProvider } as any);
    const states = ["first state", "second state", "third state"];
    const questions = {
      d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } },
    };
    const res = await a.predictBatch(states, questions);
    expect(res.length).toBe(3);
    expect(log.encoderBatches.length).toBe(1); // 1 single forward pass for all 3 states
    expect(log.encoderBatches[0].length).toBe(3); // 3 rows in inputIds
    expect(log.headCalls).toBe(1);
    for (const r of res) {
      expect((r.answers.d as ChoiceAnswer).choice).toBeDefined();
      expect(r.usage.input_tokens).toBeGreaterThan(0);
    }
  });

  it("batch result matches sequential single-state prediction", async () => {
    const dynamicProvider = {
      async runEncoder(b: { inputIds: number[][] }) {
        return { lastHidden: b.inputIds.map((row) => row.map(() => [1, 0])) };
      },
      async runHead(_h: unknown, b: { markerPos: number[][] }) {
        return {
          logits: b.markerPos.map((m) => m.map((_, i) => i + 1)),
          act: b.markerPos.map(() => [3, 0]),
        };
      },
    };
    const a = new Agent({ provider: dynamicProvider } as any);
    const states = ["ticket about refund", "urgent password reset", "general feedback"];
    const questions = {
      topic: { type: "choice", instructions: "topic?", criteria: { billing: "refunds", auth: "login", other: "misc" } },
      urgent: { type: "noul", instructions: "is urgent?" },
    };

    const sequential = await Promise.all(
      states.map((state) => a.predict(state, questions)),
    );
    const batched = await a.predictBatch(states, questions);

    expect(batched.length).toBe(sequential.length);
    for (let i = 0; i < states.length; i++) {
      expect(batched[i].model).toBe(sequential[i].model);
      const bTopic = batched[i].answers.topic as ChoiceAnswer;
      const sTopic = sequential[i].answers.topic as ChoiceAnswer;
      expect(bTopic.choice).toBe(sTopic.choice);
      expect(bTopic.probabilities).toEqual(sTopic.probabilities);
      expect(bTopic.confidence).toBe(sTopic.confidence);
      const bUrgent = batched[i].answers.urgent as NoulAnswer;
      const sUrgent = sequential[i].answers.urgent as NoulAnswer;
      expect(bUrgent.noul).toBe(sUrgent.noul);
      expect(bUrgent.confidence).toBe(sUrgent.confidence);
    }
  });

  it("chunks states into multiple forward passes when batchSize is specified", async () => {
    const log: { encoderBatches: number[][][]; headCalls: number } = { encoderBatches: [], headCalls: 0 };
    const dynamicProvider = {
      async runEncoder(b: { inputIds: number[][] }) {
        log.encoderBatches.push(b.inputIds);
        return { lastHidden: b.inputIds.map((row) => row.map(() => [1, 0])) };
      },
      async runHead(_h: unknown, b: { markerPos: number[][] }) {
        log.headCalls++;
        return {
          logits: b.markerPos.map((m) => m.map((_, i) => i + 1)),
          act: b.markerPos.map(() => [3, 0]),
        };
      },
    };
    const a = new Agent({ provider: dynamicProvider } as any);
    const states = ["s1", "s2", "s3", "s4", "s5"];
    const questions = {
      d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } },
    };
    const res = await a.predictBatch(states, questions, { batchSize: 2 });
    expect(res.length).toBe(5);
    // 5 states with batchSize 2 -> chunks of 2, 2, 1
    expect(log.encoderBatches.length).toBe(3);
    expect(log.encoderBatches[0].length).toBe(2);
    expect(log.encoderBatches[1].length).toBe(2);
    expect(log.encoderBatches[2].length).toBe(1);
    expect(log.headCalls).toBe(3);
  });

  it("rejects non-array states in predictBatch", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    await expect(a.predictBatch("single string" as any, {})).rejects.toThrow(TypeError);
  });

  it("rejects non-positive batchSize", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    await expect(a.predictBatch(["s1"], {}, { batchSize: 0 })).rejects.toThrow("batchSize must be a positive integer");
    await expect(a.predictBatch(["s1"], {}, { batchSize: -2 })).rejects.toThrow("batchSize must be a positive integer");
  });

  it("fires hooks in batch mode and records aggregated usage", async () => {
    const events: string[] = [];
    let endUsage: any = null;
    const dynamicProvider = {
      async runEncoder(b: { inputIds: number[][] }) {
        return { lastHidden: b.inputIds.map((row) => row.map(() => [1, 0])) };
      },
      async runHead(_h: unknown, b: { markerPos: number[][] }) {
        return {
          logits: b.markerPos.map((m) => m.map((_, i) => i + 1)),
          act: b.markerPos.map(() => [3, 0]),
        };
      },
    };
    const a = new Agent({
      provider: dynamicProvider,
      hooks: [
        {
          onPredictStart(_ctx: any) { events.push("start"); },
          onPredictEnd(ctx: any) {
            events.push("end");
            endUsage = ctx.usage;
          },
        },
      ],
    } as any);

    const res = await a.predictBatch(["s1", "s2"], {
      q: { type: "noul", instructions: "?" },
    });
    expect(events).toEqual(["start", "end"]);
    expect(res.length).toBe(2);
    expect(endUsage.input_tokens).toBe(res[0].usage.input_tokens + res[1].usage.input_tokens);
  });

  it("ctx.skip() short-circuits batch prediction without calling provider", async () => {
    let providerCalls = 0;
    const provider = {
      async runEncoder() { providerCalls++; return { lastHidden: [] }; },
      async runHead() { providerCalls++; return { logits: [], act: [] }; },
    };
    const cached: any = [
      { model: "cached-model", answers: { q: { type: "noul", noul: 1, confidence: 1 } }, usage: { input_tokens: 10, output_tokens: 0 } },
      { model: "cached-model", answers: { q: { type: "noul", noul: 0, confidence: 1 } }, usage: { input_tokens: 12, output_tokens: 0 } },
    ];
    let endCtx: any = null;
    const a = new Agent({
      provider: provider as any,
      hooks: [
        {
          onPredictStart(ctx: any) { ctx.skip(cached); },
          onPredictEnd(ctx: any) { endCtx = ctx; },
        },
      ],
    } as any);

    const res = await a.predictBatch(["s1", "s2"], { q: { type: "noul", instructions: "?" } });
    expect(res).toBe(cached);
    expect(providerCalls).toBe(0);
    expect(endCtx.results).toBe(cached);
    expect(endCtx.usage).toEqual({ input_tokens: 22, output_tokens: 0 });
  });

  it("evaluates mutated states when onPredictStart rewrites ctx.states", async () => {
    const encodedTexts: string[] = [];
    const tok = {
      ...defaultTokenizer(),
      encode(text: string): number[] {
        encodedTexts.push(text);
        return [1, 2, 3];
      },
    };
    const dynamicProvider = {
      async runEncoder(b: { inputIds: number[][] }) {
        return { lastHidden: b.inputIds.map((row) => row.map(() => [1, 0])) };
      },
      async runHead(_h: unknown, b: { markerPos: number[][] }) {
        return {
          logits: b.markerPos.map((m) => m.map((_, i) => i + 1)),
          act: b.markerPos.map(() => [3, 0]),
        };
      },
    };
    const a = new Agent({
      provider: dynamicProvider as any,
      tok,
      hooks: [
        {
          onPredictStart(ctx: any) {
            ctx.states = ["rewritten-a", "rewritten-b"];
          },
        },
      ],
    } as any);

    const res = await a.predictBatch(["orig-1", "orig-2"], {
      q: { type: "noul", instructions: "?" },
    });
    expect(res.length).toBe(2);
    expect(encodedTexts).toContain("rewritten-a");
    expect(encodedTexts).toContain("rewritten-b");
    expect(encodedTexts).not.toContain("orig-1");
    expect(encodedTexts).not.toContain("orig-2");
  });

  it("honors maxLen override in predictBatch", async () => {
    let capturedInputIds: number[][] = [];
    const dynamicProvider = {
      async runEncoder(b: { inputIds: number[][] }) {
        capturedInputIds = b.inputIds;
        return { lastHidden: b.inputIds.map((row) => row.map(() => [1, 0])) };
      },
      async runHead(_h: unknown, b: { markerPos: number[][] }) {
        return {
          logits: b.markerPos.map((m) => m.map((_, i) => i + 1)),
          act: b.markerPos.map(() => [3, 0]),
        };
      },
    };
    const a = new Agent({ provider: dynamicProvider as any } as any);
    const longState = "very long state text ".repeat(30);
    await a.predictBatch([longState, longState], {
      q: { type: "noul", instructions: "?" },
    }, { maxLen: 30 });

    expect(capturedInputIds.length).toBe(2);
    expect(capturedInputIds[0].length).toBeLessThanOrEqual(30);
    expect(capturedInputIds[1].length).toBeLessThanOrEqual(30);
  });
});
