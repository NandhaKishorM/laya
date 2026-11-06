import { describe, expect, it } from "vitest";
import { analyse, latinProfile, type AnalyseOptions, type LanguageDiagnosticReason } from "../src/lang.js";

const EN = "This is an English note about the order. ";
const DE = "Ich kann mich nicht anmelden und brauche Hilfe";
const PT = "Quero cancelar a minha conta porque não consigo entrar";

function explained(state: unknown) {
  const plain = analyse(state);
  expect(analyse(state, { explain: false })).toEqual(plain);
  expect(plain).not.toHaveProperty("diagnostic");
  const { diagnostic, ...legacy } = analyse(state, { explain: true });
  expect(legacy).toEqual(plain);
  return diagnostic;
}

describe("language diagnostics", () => {
  it.each<[string, LanguageDiagnosticReason]>([
    ["123 !!!", "no_letters"],
    ["请帮我取消这个订单并且退款给我", "non_latin_script"],
    ["Turn lights off", "insufficient_words"],
    ["glömt lösenord", "short_swedish"],
    ["Please refund the duplicate charge", "language_evidence"],
    ["xyzzy frobnitz quux blorb", "no_language_evidence"],
    ["MON DES EST LA", "caps_veto"],
    ["MON DES EST LA refund", "caps_filtered"],
    ["və the quick brown", "non_english_markers"],
    ["WIE SPÄT IST ES IN KÖLN", "non_english_markers"],
  ])("records the decisive whole-state branch for %s", (state, reason) => {
    expect(explained(state)).toEqual({ stage: "whole_state", reason, source: null });
  });

  it("does not add metadata to the public Latin profile", () => {
    expect(Object.keys(latinProfile(DE))).toEqual(["language", "englishHits", "diacriticRate", "looksNonEnglish"]);
  });

  it("locates a nested segment, with zero-based leaves and one-based lines", () => {
    expect(explained({ note: EN.repeat(20), payload: { messages: [null, 42, `status\r\n\r\n${DE}`, PT] } }))
      .toEqual({ stage: "segment", reason: "named_segment", source: { leafIndex: 1, path: ["payload", "messages", 2], line: 3 } });
  });

  it("locates a root multiline string with an empty path", () => {
    expect(explained(`${EN.repeat(20)}\n${DE}`))
      .toEqual({ stage: "segment", reason: "named_segment", source: { leafIndex: 0, path: [], line: 2 } });
    expect(explained(new TextEncoder().encode(`${EN.repeat(20)}\n${DE}`))?.source)
      .toEqual({ leafIndex: 0, path: [], line: 2 });
  });

  it("keeps the first duplicate segment rather than searching the input by value", () => {
    expect(explained({ note: EN.repeat(20), first: DE, second: DE })?.source)
      .toEqual({ leafIndex: 1, path: ["first"], line: 1 });
  });

  it.each<[string, LanguageDiagnosticReason]>([
    ["ich kann\nmich nicht", "named_value"],
    ["请帮我取消这个订单并且退款给我", "non_latin_value"],
    ["The naïve façade needs a fresh coat of paint", "diacritic_value"],
  ])("reports the existing whole-leaf fallback for %s", (body, reason) => {
    const state = { note: EN.repeat(140), body };
    expect(explained(state)).toEqual({ stage: "value", reason, source: { leafIndex: 1, path: ["body"], line: null } });
    expect(analyse(state, { explain: true })).toMatchObject({ script: "latin", scriptProfile: { latin: 1 }, mixedSegment: null });
  });

  it("attributes the longest eligible value and keeps the first on a tie", () => {
    expect(explained({ note: EN.repeat(140), short: "ich kann\nmich nicht", longer: DE, tie: DE.toLowerCase() })?.source)
      .toEqual({ leafIndex: 2, path: ["longer"], line: null });
  });

  it("counts only decoded string leaves and preserves sparse-array indices", () => {
    const items = [];
    items[2] = new Uint8Array([0xff]);
    items[5] = new TextEncoder().encode(DE);
    expect(explained({ note: EN.repeat(20), items })?.source)
      .toEqual({ leafIndex: 1, path: ["items", 5], line: 1 });
  });

  it("keeps numeric object keys as strings", () => {
    expect(explained({ note: EN.repeat(140), payload: { "2": DE } })?.source)
      .toEqual({ leafIndex: 1, path: ["payload", "2"], line: null });
  });

  it.each(["x", "😀"])("bounds %s keys in Unicode code points and omits the entire oversized path", (char) => {
    expect(explained({ note: EN.repeat(140), [char.repeat(128)]: DE })?.source?.path).toEqual([char.repeat(128)]);
    const diagnostic = explained({ note: EN.repeat(140), [char.repeat(129)]: { body: DE } });
    expect(diagnostic?.source).toEqual({ leafIndex: 1, path: null, line: null });
    expect(JSON.stringify(diagnostic)).not.toContain(char.repeat(129));
    expect(JSON.stringify(diagnostic)).not.toContain(DE);
  });

  it("retains the depth-six traversal boundary", () => {
    const nest = (depth: number): unknown => depth ? { child: nest(depth - 1) } : DE;
    expect(explained({ note: EN.repeat(140), body: nest(5) })?.source)
      .toEqual({ leafIndex: 1, path: ["body", ...Array(5).fill("child")], line: null });
    expect(explained({ note: EN.repeat(140), body: nest(6) })?.stage).toBe("whole_state");
  });

  it("retains the upstream code-point window and the root-string fallback boundary", () => {
    expect(explained("😀".repeat(2000) + DE))
      .toEqual({ stage: "whole_state", reason: "language_evidence", source: null });
    expect(explained({ padding: "😀".repeat(2000), body: DE }))
      .toEqual({ stage: "whole_state", reason: "language_evidence", source: null });
    expect(explained("😀".repeat(4000) + DE)).toEqual({ stage: "whole_state", reason: "no_letters", source: null });
    expect(explained({ padding: "😀".repeat(4000), body: DE }))
      .toEqual({ stage: "value", reason: "named_value", source: { leafIndex: 1, path: ["body"], line: null } });
    expect(explained(EN.repeat(140) + "\n" + DE)?.stage).toBe("whole_state");
  });

  it("preserves eager object getter order and captures the decisive traversal", () => {
    const run = (opts?: AnalyseOptions) => {
      const events: string[] = [];
      let reads = 0;
      const state = {
        get note() { events.push("note"); return EN.repeat(140); },
        get payload() {
          events.push("payload");
          const key = ++reads === 4 ? "winner" : "earlier";
          return { get [key]() { events.push(key); return DE; } };
        },
        get tail() { events.push("tail"); return null; },
      };
      return { result: analyse(state, opts), events };
    };
    const plain = run();
    expect(run({ explain: false })).toEqual(plain);
    const on = run({ explain: true });
    const { diagnostic, ...legacy } = on.result;
    expect(legacy).toEqual(plain.result);
    expect(on.events).toEqual(plain.events);
    expect(on.events).toEqual(["note", "payload", "tail", "earlier", "note", "payload", "tail", "earlier",
      "note", "payload", "tail", "earlier", "note", "payload", "tail", "winner"]);
    expect(diagnostic?.source).toEqual({ leafIndex: 1, path: ["payload", "winner"], line: null });
  });

  it("does not infer indices for custom array iterators or read their getters twice", () => {
    const run = (opts?: AnalyseOptions) => {
      const events: string[] = [];
      const values = [DE, EN.repeat(140)];
      const iterator = function* (this: string[]) {
        expect(this).toBe(values);
        events.push("start");
        yield this[1];
        yield this[0];
        events.push("end");
      };
      // for-of invokes the method directly; an own .call property must not intercept it.
      Object.defineProperty(iterator, "call", { value() { throw new Error("not the iteration protocol"); } });
      Object.defineProperty(values, Symbol.iterator, {
        get() {
          events.push("iterator");
          return iterator;
        },
      });
      return { result: analyse({ values }, opts), events };
    };
    const plain = run();
    const on = run({ explain: true });
    const { diagnostic, ...legacy } = on.result;
    expect(legacy).toEqual(plain.result);
    expect(on.events).toEqual(plain.events);
    expect(diagnostic?.source).toEqual({ leafIndex: 1, path: null, line: null });
  });

  it.each([false, true])("closes a custom iterator if a child getter throws (explain=%s)", (explain) => {
    let closed = 0;
    let passes = 0;
    const failure = new Error("getter failed");
    const values: unknown[] = [];
    values[Symbol.iterator] = function* () {
      passes++;
      try {
        yield EN.repeat(140);
        yield { get body() { if (passes === 4) throw failure; return DE; } };
      } finally { closed++; }
      return undefined;
    };
    expect(() => analyse(values, { explain })).toThrow(failure);
    expect(passes).toBe(4);
    expect(closed).toBe(4);
  });

  it.each([null, 0, 1, "true", {}, []])("rejects a non-boolean explain value %j before reading state", (explain) => {
    const state = { get body() { throw new Error("must not read"); } };
    expect(() => analyse(state, { explain } as AnalyseOptions)).toThrow("explain must be a boolean");
  });
});
