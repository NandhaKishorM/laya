import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Agent, Router, OptionStabilityHook, PredictContext, clearDefaultHooks, defaultTokenizer,
  setDefaultHooks, calibrateReliability, decideReliability, type Batch, type QuestionDef, type SessionProvider,
} from "../src/index.js";

afterEach(clearDefaultHooks);

// Exercise real validation, encoding, collation, decoding and hook dispatch. Only the
// model provider is synthetic: it selects the option span that still contains "approve".
function semanticAgent(extra: Record<string, unknown> = {}) {
  const batches: Batch[] = [];
  const tok = {
    ...defaultTokenizer(),
    encode(text: string) {
      return text.split(/[^\p{L}\p{N}_]+/u).filter(Boolean).map((word) => word === "approve" ? 900 : 901);
    },
  };
  const provider: SessionProvider = {
    async runEncoder(batch) {
      batches.push(batch);
      await Promise.resolve();
      return { lastHidden: batch.inputIds.map(() => [[0]]) };
    },
    async runHead(_hidden, batch) {
      const logits = batch.markerPos.map((positions, row) => positions.map((start, i) => {
        if (!batch.markerMask[row][i]) return -100;
        const next = positions[i + 1] || batch.inputIds[row].length;
        const sep = batch.inputIds[row].indexOf(tok.sepId, start);
        const end = sep < 0 ? next : Math.min(sep, next);
        return batch.inputIds[row].slice(start, end).includes(900) ? 6 : 0;
      }));
      return { logits, act: batch.qtype.map(() => [3, 0]) };
    },
  };
  return { agent: new Agent({ provider, tok, ...extra }), batches };
}

const hook = () => new OptionStabilityHook({ accept: 0.8, escalate: 0.6, variants: "fast", seed: "shared" });
const question = (criteria: unknown, order?: number[]): Record<string, QuestionDef> => ({
  action: { type: "choice", instructions: "Choose action", criteria, ...(order ? { option_order: order } : {}) },
});

describe("OptionStabilityHook through the local runtime", () => {
  it("requires explicit finite thresholds for JavaScript callers too", () => {
    for (const options of [{}, { accept: null }, { accept: undefined, escalate: null },
      { accept: NaN, escalate: 0 }, { accept: 0.5, escalate: 0.6 }, { accept: true, escalate: null }]) {
      expect(() => new OptionStabilityHook(options as never)).toThrow();
    }
    expect(() => decideReliability(0.9, undefined as never)).toThrow();
    expect(decideReliability(0.9, null, null)).toBeNull();
    expect(decideReliability(0.5, 0.8, null)).toBe("VERIFY");
  });

  it("rejects incomplete or invalid calibration evidence instead of silently dropping rows", () => {
    expect(() => calibrateReliability([0.8, 0.9], [true])).toThrow();
    expect(() => calibrateReliability([0.8], [true, false])).toThrow();
    expect(() => calibrateReliability([], [])).toThrow();
    expect(() => calibrateReliability([0.8], [1] as never)).toThrow();
    for (const score of [NaN, Infinity, -0.1, 1.1]) {
      expect(() => calibrateReliability([score], [true])).toThrow();
    }
    expect(() => calibrateReliability([0.8], [true], { targetAccuracy: NaN })).toThrow();
    expect(() => calibrateReliability([0.8], [true], { maxEscalateAccuracy: 1.1 })).toThrow();
  });

  it("cleans incomplete or malformed probe results without manufacturing reliability", () => {
    for (const bad of [undefined, null, { type: "choice", probabilities: { A: 1 } },
      { type: "choice", choice: "A", probabilities: { A: NaN } },
      { type: "choice", choice: "A", probabilities: { A: 1.1 } }]) {
      const stability = hook();
      const ctx = new PredictContext({ states: ["state"], questions: question(["approve", "reject"]) });
      stability.onPredictStart(ctx);
      const primary = { type: "choice", choice: "approve", probabilities: { approve: 1 } };
      const answers = { action: primary, "action::rel1": bad, "action::rel2": bad };
      ctx.results = [{ answers, usage: { options: { action: {}, "action::rel1": {} },
        truncated_questions: ["action", "action::rel2"] } }];
      expect(() => stability.onPredictEnd(ctx)).not.toThrow();
      expect(answers).toEqual({ action: primary });
      expect(primary).not.toHaveProperty("reliability");
      expect(ctx.results[0].usage).toEqual({ options: { action: {} }, truncated_questions: ["action"] });
    }
  });

  it.each([
    ["list", ["approve", "reject", "wait"]],
    ["empty descriptions", { approve: null, reject: "", wait: "  " }],
    ["mixed", { approve: "approve request", reject: null, wait: { action: "wait" } }],
  ])("preserves %s semantics and caller inputs in a chunked batch", async (_name, criteria) => {
    const q = question(criteria, [2, 0, 1]);
    const original = structuredClone(q);
    const events: string[] = [];
    const { agent, batches } = semanticAgent({ hooks: [hook()] });
    const results = await agent.predictBatch(["first", "second"], q, {
      batchSize: 1,
      async onPredictStart() { await Promise.resolve(); events.push("start"); },
      async onPredictEnd(ctx) {
        await Promise.resolve();
        expect(Object.keys(ctx.results![0].answers as object)).toEqual(["action"]);
        events.push("end");
      },
    });
    expect(events).toEqual(["start", "end"]);
    expect(q).toEqual(original);
    expect(batches.map((b) => b.qtype.length)).toEqual([3, 3]);
    for (const result of results) {
      expect(Object.keys(result.answers)).toEqual(["action"]);
      expect(result.answers.action).toMatchObject({ choice: "approve", reliability: {
        decision: "ACCEPT", stability: 1, distinct_choices: ["approve"], n_variants: 3,
        variant_choices: { original: "approve", reversed: "approve", letters_reversed: "approve" },
      } });
      expect(result.usage.truncated_questions?.some((id) => id.includes("::rel"))).toBe(false);
    }
  });

  it("physically permutes integer-like labels and maps through caller option_order", async () => {
    const stability = new OptionStabilityHook({ accept: null, escalate: null, variants: ["reversed"], seed: "n" });
    const { agent, batches } = semanticAgent({ hooks: [stability] });
    const result = await agent.predict("state", question({ "1": "reject", "2": "approve", "10": "wait" }, [1, 2, 0]));
    expect(result.answers.action).toMatchObject({ choice: "2", reliability: { stability: 1, n_variants: 2 } });
    // Original physical order is approve, wait, reject; reversed+caller order is approve,
    // reject, wait. Non-semantic tokens coincide in the tokenizer, so inspect transformed
    // questions separately below as well as checking the actual decoder's label.
    expect(batches[0].markerPos).toHaveLength(2);
    const ctx = new PredictContext({ states: ["state"], questions: question({ "1": "reject", "2": "approve", "10": "wait" }) });
    stability.onPredictStart(ctx);
    const variant = ctx.questions["action::rel1"] as QuestionDef;
    expect(variant.option_order).toEqual([2, 1, 0]);
    stability.onError(ctx);
  });

  it("keeps concurrent calls isolated when a hook instance is shared", async () => {
    const stability = hook();
    const { agent } = semanticAgent({ hooks: [stability] });
    const [a, b] = await Promise.all([
      agent.predict("one", question(["approve", "reject"])),
      agent.predictBatch(["two", "three"], question(["wait", "approve", "reject"])),
    ]);
    for (const result of [a, ...b]) {
      expect(result.answers.action).toMatchObject({ choice: "approve", reliability: { stability: 1 } });
      expect(Object.keys(result.answers)).toEqual(["action"]);
    }
  });

  it("does not manufacture probe evidence for a cached primary-only answer", async () => {
    const stability = hook();
    const { agent, batches } = semanticAgent({ hooks: [stability] });
    const cached = { answers: { action: { type: "choice", choice: "approve", probabilities: { approve: 1 } } }, usage: {} };
    const result = await agent.predict("cached", question(["approve", "reject"]), {
      onPredictStart(ctx) { ctx.skip([structuredClone(cached)]); },
    });
    expect(result.answers.action).not.toHaveProperty("reliability");
    expect(batches).toHaveLength(0);
    expect((await agent.predict("fresh", question(["approve", "reject"]))).answers.action)
      .toHaveProperty("reliability.stability", 1);
  });

  it.each(["provider", "validation", "timeout"])("cleans the plan after a %s failure and preserves the error", async (mode) => {
    const stability = hook();
    let captured: PredictContext | undefined;
    const { agent } = semanticAgent({ hooks: [{ onPredictStart(ctx: PredictContext) { captured = ctx; } }, stability] });
    if (mode === "provider") vi.spyOn(agent.provider, "runEncoder").mockRejectedValueOnce(new Error("provider failed"));
    const q = question(["approve", "reject"]);
    if (mode === "validation") q.action.option_order = [0, 0];
    const opts = mode === "timeout" ? {
      hooksTimeout: 0.001, onPredictStart: () => new Promise<void>(() => {}),
    } : {};
    await expect(agent.predict("state", q, opts)).rejects.toThrow(
      mode === "provider" ? /provider failed/ : mode === "validation" ? /option_order/ : /exceeded/,
    );
    expect(captured!.error).not.toBeNull();
    // Calling end again on the captured failed context must not find an old run plan.
    const answers = Object.fromEntries(Object.entries(captured!.questions).map(([qid, definition]) => {
      const criteria = (definition as QuestionDef).criteria;
      const key = Array.isArray(criteria) ? criteria[0] : Object.keys(criteria as object)[0];
      return [qid, { type: "choice", choice: key, probabilities: { [key]: 1 } }];
    }));
    captured!.results = [{ answers, usage: {} }];
    stability.onPredictEnd(captured!);
    expect(answers.action).not.toHaveProperty("reliability");
    expect((await agent.predict("recovered", question(["approve", "reject"]))).answers.action)
      .toHaveProperty("reliability.stability", 1);
  });

  it.each([false, true])("runs a default stability hook once through Router batch, fallback=%s", async (fallback) => {
    const { agent } = semanticAgent();
    const loaded = fallback ? { systemOne: agent.systemOne.bind(agent), cfg: agent.cfg } : agent;
    const router = new Router({ default: "english", loader: (() => loaded) as never });
    setDefaultHooks(hook());
    const results = await router.predictBatch([
      { state: "one", questions: question(["approve", "reject"]) },
      { state: "two", questions: question(["approve", "reject"]) },
    ]);
    for (const result of results) {
      expect(result.answers.action).toMatchObject({ choice: "approve", reliability: { stability: 1 } });
      expect(Object.keys(result.answers)).toEqual(["action"]);
    }
  });
});
