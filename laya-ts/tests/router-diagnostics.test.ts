import { describe, expect, it, vi } from "vitest";
import { Router, type RouteOptions, type RouteDecision } from "../src/router.js";
import { analyse, type AnalyseOptions } from "../src/lang.js";
import type { PredictContext } from "../src/hooks.js";

const STATE = {
  note: "This is an English note about the order. ".repeat(140),
  customer: "Ich kann mich nicht anmelden und brauche Hilfe",
};
const QUESTIONS = { intent: { type: "noul", instructions: "Is a refund requested?" } };
const TYPED = Object.fromEntries(["action", "needs_review", "outcome", "risk", "urgency"]
  .map((key) => [key, { type: "noul", instructions: "?" }]));

describe("Router language diagnostics", () => {
  it("adds only detection.diagnostic, without loading a checkpoint", () => {
    const loader = vi.fn(async () => { throw new Error("must not load"); });
    const router = new Router({ loader });
    const plain = router.route(STATE, QUESTIONS);
    expect(router.route(STATE, QUESTIONS, { explain: false })).toEqual(plain);
    expect(plain.detection).not.toHaveProperty("diagnostic");
    const on = router.route(STATE, QUESTIONS, { explain: true });
    expect(on.detection).toEqual(analyse(STATE, { explain: true }));
    const { diagnostic, ...detection } = on.detection!;
    expect({ ...on, detection }).toEqual(plain);
    expect(diagnostic).toEqual({ stage: "value", reason: "named_value", source: { leafIndex: 1, path: ["customer"], line: null } });
    expect(loader).not.toHaveBeenCalled();
    expect(router.loaded).toEqual([]);
  });

  it.each<RouteOptions>([
    { model: "english", task: "multilingual", lang: "de" },
    { task: "english", lang: "de" },
    { lang: "en" },
    { langGuess: "de" },
    { lang_guess: () => "de" },
  ])("preserves override precedence and leaves detection null for %j", (opts) => {
    const state = { get body() { throw new Error("detector must not read"); } };
    const router = new Router();
    const plain = router.route(state, QUESTIONS, opts);
    const on = router.route(state, QUESTIONS, { ...opts, explain: true });
    expect(on).toEqual(plain);
    expect(on.detection).toBeNull();
  });

  it("keeps typed workflows ahead of language hints and detection", () => {
    const hint = vi.fn(() => "en");
    const router = new Router({ langGuess: hint, autoTaskDetection: true });
    const state = { get body() { throw new Error("must not read"); } };
    const on = router.route(state, TYPED, { explain: true, lang: "en" });
    expect(on.model).toBe("typed-decisions");
    expect(on.workflow).not.toBeNull();
    expect(on.detection).toBeNull();
    expect(hint).not.toHaveBeenCalled();
  });

  it("calls abstaining hints once, then exposes the detector result", () => {
    const perCall = vi.fn(() => null);
    const installed = vi.fn(() => null);
    const router = new Router({ langGuess: installed });
    const on = router.route(STATE, QUESTIONS, { explain: true, lang: " ", langGuess: perCall });
    expect(on.detection?.diagnostic?.stage).toBe("value");
    expect(perCall).toHaveBeenCalledExactlyOnceWith(STATE);
    expect(installed).toHaveBeenCalledExactlyOnceWith(STATE);
    const decisive = new Router({ langGuess: () => "de" }).route(STATE, QUESTIONS, { explain: true });
    expect(decisive.detection).toBeNull();
  });

  it("exposes diagnostics to onRoute and preserves its final replacement", () => {
    const replacement: RouteDecision = {
      model: "english", repo: "convaiinnovations/laya/english", reason: "caller policy", detection: null, workflow: null,
    };
    const events: string[] = [];
    const router = new Router({ hooks: [{ onRoute(ctx: PredictContext) {
      expect((ctx.decision as unknown as RouteDecision).detection?.diagnostic?.reason).toBe("named_value");
      events.push("installed");
    } }] });
    const result = router.route(STATE, QUESTIONS, { explain: true, hooks: [{ onRoute(ctx: PredictContext) {
      events.push("call");
      ctx.decision = replacement as unknown as Record<string, unknown>;
    } }] });
    expect(events).toEqual(["installed", "call"]);
    expect(result).toBe(replacement);
    expect(result.detection).toBeNull();
  });

  it("routes a heterogeneous batch in order and passes explain to each hook", () => {
    const router = new Router();
    const requests = [
      { state: STATE, questions: QUESTIONS },
      { state: STATE, questions: QUESTIONS, lang: "en" },
      { state: "123 !!!", questions: QUESTIONS },
    ];
    const seen: unknown[] = [];
    const on = router.routeBatch(requests, { explain: true, hooks: [{ onRoute(ctx: PredictContext) {
      seen.push((ctx.decision as unknown as RouteDecision).detection?.diagnostic ?? null);
    } }] });
    expect(on).toEqual(requests.map((request) => router.route(request.state, request.questions, { ...request, explain: true })));
    expect(seen).toEqual([analyse(STATE, { explain: true }).diagnostic, null, { stage: "whole_state", reason: "no_letters", source: null }]);
    const plain = router.routeBatch(requests);
    expect(router.routeBatch(requests, { explain: false })).toEqual(plain);
    for (const decision of plain) if (decision.detection) expect(decision.detection).not.toHaveProperty("diagnostic");
    expect(router.routeBatch([], { explain: true })).toEqual([]);
    expect(router.loaded).toEqual([]);
  });

  it.each([null, 0, 1, "true", {}, []])("rejects invalid explain %j even for overrides or an empty batch", (value) => {
    const explain = value as AnalyseOptions["explain"];
    const hook = vi.fn();
    const router = new Router({ hooks: [{ onRoute: hook }] });
    expect(() => router.route(STATE, QUESTIONS, { model: "english", explain })).toThrow("explain must be a boolean");
    expect(() => router.route(STATE, TYPED, { explain })).toThrow("explain must be a boolean");
    expect(() => router.routeBatch([], { explain })).toThrow("explain must be a boolean");
    expect(() => router.routeBatch([{ state: STATE, questions: QUESTIONS, lang: "en" }], { explain }))
      .toThrow("explain must be a boolean");
    expect(hook).not.toHaveBeenCalled();
  });
});
