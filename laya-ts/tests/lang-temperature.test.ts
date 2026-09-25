import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import { Router } from "../src/router.js";

// Same fixed logit row as agent.test.ts: softmax([2, 0]) at temperature 1 is
// [0.8808, 0.1192]; at 2 -> [0.7311, 0.2689]; at 5 -> [0.5987, 0.4013].
const fakeProvider = () => ({
  async runEncoder(_b: any) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: any) { return { logits: [[2, 0]], act: [[3, 0]] }; },
});

const CHOICE = { d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } } } as any;
const NOUL = { d: { type: "noul", instructions: "q?" } } as any;

describe("lang_temperatures", () => {
  it("applies the matching language override, normalising both sides to the base subtag", async () => {
    const a = new Agent({
      provider: fakeProvider(),
      lang_temperatures: { "de-AT": { temperature: [2, 2, 2] } },
    } as any);
    expect(Object.keys(a.langTemperatures)).toEqual(["de"]);
    const r: any = await a.systemOne("hi", CHOICE, { lang: "de" });
    expect(r.answers.d.probabilities.x).toBeCloseTo(0.7311, 4);
    const regional: any = await a.systemOne("hi", CHOICE, { lang: "de-CH" });
    expect(regional.answers.d.probabilities.x).toBeCloseTo(0.7311, 4);
    const batched: any = await a.predictBatch(["hi"], CHOICE, { lang: "de" });
    expect(batched[0].answers.d.probabilities.x).toBeCloseTo(0.7311, 4);
  });

  it("no lang or an unknown lang keeps the base temperature", async () => {
    const a = new Agent({
      provider: fakeProvider(),
      lang_temperatures: { de: { temperature: [2, 2, 2] } },
    } as any);
    const plain: any = await a.systemOne("hi", CHOICE);
    expect(plain.answers.d.probabilities.x).toBeCloseTo(0.8808, 4);
    const other: any = await a.systemOne("hi", CHOICE, { lang: "fr" });
    expect(other.answers.d.probabilities.x).toBeCloseTo(0.8808, 4);
  });

  it("an override without buckets ignores the base per-bucket overrides", async () => {
    // Python parity: a matching lang override replaces the scale wholesale, so the base
    // temperature_by_options entry must NOT leak into the overridden language.
    const a = new Agent({
      provider: fakeProvider(),
      temperature_by_options: { "choice:2": 2.0 },
      lang_temperatures: { de: { temperature: [1, 1, 1] } },
    } as any);
    const base: any = await a.systemOne("hi", CHOICE);
    expect(base.answers.d.probabilities.x).toBeCloseTo(0.7311, 4);
    const german: any = await a.systemOne("hi", CHOICE, { lang: "de" });
    expect(german.answers.d.probabilities.x).toBeCloseTo(0.8808, 4);
  });

  it("an override with only buckets inherits the base raw temperature per type", async () => {
    const a = new Agent({
      provider: fakeProvider(),
      lang_temperatures: { de: { temperature_by_options: { "choice:2": 2.0 } } },
    } as any);
    const choice: any = await a.systemOne("hi", CHOICE, { lang: "de" });
    expect(choice.answers.d.probabilities.x).toBeCloseTo(0.7311, 4);
    // noul has no bucket override -> the inherited base temperature (1.0) applies.
    const noul: any = await a.systemOne("hi", NOUL, { lang: "de" });
    expect(noul.answers.d.noul).toBeCloseTo(0.1192, 4);
  });

  it("clamps override values like the base temperature", async () => {
    const a = new Agent({
      provider: fakeProvider(),
      lang_temperatures: { de: { temperature: [100, 0, 1] } },
    } as any);
    expect(a.langTemperatures.de.temperature).toEqual([5, 0.5, 1]);
    const r: any = await a.systemOne("hi", CHOICE, { lang: "de" });
    expect(r.answers.d.probabilities.x).toBeCloseTo(0.5987, 4);
  });

  it("rejects an override temperature that is not a list of 3 floats", () => {
    expect(
      () => new Agent({ provider: fakeProvider(), lang_temperatures: { de: { temperature: [1, 1] } } } as any),
    ).toThrow('Language override "de" temperature must be a list of 3 floats');
    expect(
      () => new Agent({ provider: fakeProvider(), lang_temperatures: { de: { temperature: 2 } } } as any),
    ).toThrow('Language override "de" temperature must be a list of 3 floats');
  });

  it("a null entry is an empty override (base raw temperature, no buckets)", async () => {
    const a = new Agent({
      provider: fakeProvider(),
      temperature_by_options: { "choice:2": 2.0 },
      lang_temperatures: { de: null },
    } as any);
    const r: any = await a.systemOne("hi", CHOICE, { lang: "de" });
    expect(r.answers.d.probabilities.x).toBeCloseTo(0.8808, 4);
  });
});

describe("Router.predict lang forwarding", () => {
  const questions = { q: { type: "noul", instructions: "?" } } as any;

  function routerRecording(seen: any[]) {
    const stub = {
      async systemOne(_s: any, _q: any, opts?: any) {
        seen.push(opts);
        return { model: "m", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
      },
    };
    return new Router({ loader: () => stub } as any);
  }

  it("forwards an explicit lang, winning over detection", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen);
    // This romanized Bangla text detects as "bn"; the explicit lang must win.
    await r.predict("ami ekta ticket khulsi, kalke theke payment hocche na", questions, { lang: "de" });
    expect(seen).toEqual([{ lang: "de" }]);
  });

  it("forwards the detected language when no explicit lang is given", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen);
    await r.predict("ami ekta ticket khulsi, kalke theke payment hocche na", questions);
    expect(seen).toEqual([{ lang: "bn" }]);
  });

  it("forwards null when nothing non-English was detected", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen);
    await r.predict("please refund my ticket", questions);
    expect(seen).toEqual([{ lang: null }]);
  });

  it("an explicit lang is forwarded verbatim, even \"en\"", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen);
    await r.predict("please refund my ticket", questions, { lang: "en" });
    expect(seen).toEqual([{ lang: "en" }]);
  });
});

describe("Router.predictBatch lang forwarding", () => {
  const questions = { q: { type: "noul", instructions: "?" } } as any;
  const req = (state: unknown, overrides: Record<string, unknown> = {}) =>
    ({ state, questions, ...overrides }) as any;

  // Records every agent.predictBatch call. `langTemperatures` is what gates the group split,
  // exactly as on Agent: an agent without overrides keeps sharing one forward pass.
  function routerRecording(seen: any[], langTemperatures: Record<string, unknown> = { de: {} }) {
    const stub = {
      langTemperatures,
      async predictBatch(states: unknown[], _q: any, opts?: any) {
        seen.push({ states, opts });
        return states.map((state) => ({ model: "m", answers: { seen: state }, usage: {} }));
      },
      async systemOne(state: unknown, _q: any, opts?: any) {
        seen.push({ states: [state], opts });
        return { model: "m", answers: { seen: state }, usage: {} };
      },
    };
    return new Router({ loader: () => stub } as any);
  }

  it("forwards an explicit lang to agent.predictBatch, winning over detection", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen);
    // This romanized Bangla text detects as "bn"; the explicit lang must win.
    await r.predictBatch([req("ami ekta ticket khulsi, kalke theke payment hocche na", { lang: "de" })]);
    expect(seen.map((c) => c.opts.lang)).toEqual(["de"]);
  });

  it("forwards the detected language when no explicit lang is given", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen);
    await r.predictBatch([req("ami ekta ticket khulsi, kalke theke payment hocche na")]);
    expect(seen.map((c) => c.opts.lang)).toEqual(["bn"]);
  });

  it("forwards no lang when nothing non-English was detected", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen);
    await r.predictBatch([req("please refund my ticket")]);
    expect(seen.map((c) => c.opts.lang)).toEqual([undefined]);
  });

  it("does not share a forward pass across languages", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen);
    await r.predictBatch([
      req("hallo", { lang: "de", model: "english" }),
      req("buna", { lang: "ro", model: "english" }),
      req("hallo zwei", { lang: "de", model: "english" }),
    ]);
    expect(seen.map((c) => [c.opts.lang, c.states])).toEqual([
      ["de", ["hallo", "hallo zwei"]],
      ["ro", ["buna"]],
    ]);
  });

  it("keeps one shared batch when the agent carries no lang overrides", async () => {
    const seen: any[] = [];
    const r = routerRecording(seen, {});
    await r.predictBatch([
      req("hallo", { lang: "de", model: "english" }),
      req("hello", { lang: "fr", model: "english" }),
    ]);
    expect(seen.map((c) => [c.opts.lang, c.states])).toEqual([[undefined, ["hallo", "hello"]]]);
  });

  it("applies the same lang override on the batch path as predict (German override)", async () => {
    // The reported gap: with a German override, predict returned 0.7311 and predictBatch
    // 0.8808, because the batched path never forwarded the request's language.
    const agent = new Agent({
      provider: fakeProvider(),
      lang_temperatures: { de: { temperature: [2, 2, 2] } },
    } as any);
    const r = new Router({ loader: () => agent } as any);
    const single: any = await r.predict("hallo", CHOICE, { lang: "de" });
    const batch: any[] = await r.predictBatch([{ state: "hallo", questions: CHOICE, lang: "de" } as any]);
    expect(single.answers.d.probabilities.x).toBeCloseTo(0.7311, 4);
    expect(batch[0].answers.d.probabilities.x).toBeCloseTo(0.7311, 4);
  });
});
