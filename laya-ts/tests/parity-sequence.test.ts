// laya-ts/tests/parity-sequence.test.ts
import { describe, expect, it } from "vitest";
import { renderOptions, serializeState, buildSequence, collateItems, type InternalQ } from "../src/common.js";
import type { TokenizerLike } from "../src/tokenizer.js";
describe("common parity", () => {
  it("renders choice with JSON criteria", () => {
    expect(renderOptions({ t: "choice", ins: "x", crit: { a: "yes", b: null } }))
      .toEqual(["a: yes", "b"]);
  });
  it("renders noul defaults", () => {
    expect(renderOptions({ t: "noul", ins: "x", crit: null })[0].startsWith("false: ")).toBe(true);
  });
  it("builds [CLS] head [SEP] markers [SEP] state [SEP]", () => {
    const tok = { clsId: 101, sepId: 102, maskId: 103, maskToken: "[MASK]", encode(s: string): number[] {
      return s.split(/\s+/).filter(Boolean).map((_, i) => 1000 + i); } };
    const { ids, markers } = buildSequence(tok as any,
      "hi", { t: "choice", ins: "pick", crit: { a: "x", b: "y" } }, 64, 32);
    expect(ids[0]).toBe(101);
    expect(markers.length).toBe(2);
    expect(ids[ids.length - 1]).toBe(102);
  });
  it("serializes dict state as JSON", () => {
    expect(serializeState({ body: "x" })).toBe('{"body": "x"}');
  });
  it("formats non-integer numbers like Python's float repr", () => {
    // Expected strings are Python's json.dumps({"x": v}, ensure_ascii=False). Python switches
    // to exponent notation below 1e-4 and pads the exponent to two digits; JavaScript's
    // Number#toString switches only below 1e-6 and never pads, so the model read other tokens.
    const cases: [number, string][] = [
      [0.5, '{"x": 0.5}'],
      [0.0001, '{"x": 0.0001}'],
      [0.00005, '{"x": 5e-05}'],
      [1.25e-5, '{"x": 1.25e-05}'],
      [3e-7, '{"x": 3e-07}'],
      [-2.5e-8, '{"x": -2.5e-08}'],
      [1.5e-10, '{"x": 1.5e-10}'],
      [5e-324, '{"x": 5e-324}'],
    ];
    for (const [v, want] of cases) expect(serializeState({ x: v })).toBe(want);
    expect(renderOptions({ t: "score", ins: "x", crit: [0.00005] })).toEqual(["level 0: 5e-05"]);
    // an integer-valued number reads as a Python int, which keeps plain digits
    expect(serializeState({ n: 49 })).toBe('{"n": 49}');
    expect(serializeState({ n: 1e16 })).toBe('{"n": 10000000000000000}');
  });
  it("truncateLeft keeps tail of state (py parity)", () => {
    const tok = {
      clsId: 101, sepId: 102, maskId: 103, maskToken: "[MASK]",
      encode(s: string): number[] { return s.split(/\s+/).filter(Boolean).map((_, i) => 1000 + i); },
    } as any;
    const q = { t: "choice", ins: "pick", crit: { a: "x", b: "y" } } as any;
    const state = Array.from({ length: 20 }, (_, i) => `w${i}`).join(" ");
    const a = buildSequence(tok, state, q, 16, 8);
    const b = buildSequence(tok, state, q, 16, 8, undefined, true);
    expect(a.ids.length).toBeLessThanOrEqual(16);
    expect(b.ids.length).toBeLessThanOrEqual(16);
    expect(a.ids).not.toEqual(b.ids);
    // room=3: head keeps [1000,1001,1002], tail keeps [1017,1018,1019]
    expect(a.ids.slice(12, 15)).toEqual([1000, 1001, 1002]);
    expect(b.ids.slice(12, 15)).toEqual([1017, 1018, 1019]);
  });
  // With no room left for the state, `slice(-0)` kept all of it: the closing [SEP] was replaced by the
  // *first* state token, i.e. the wrong end of the state and an unterminated sequence.
  it.each<[number, string[]]>([
    [0, []],
    [2, ["two", "three"]],
    [4, ["one", "two", "three"]],
    [10, ["one", "two", "three"]],
  ])("truncate_left/room=%d keeps the tail", (room, kept) => {
    // same stub as Python's _SeqTok: one token per word, ids assigned in first-seen order
    const vocab = new Map<string, number>();
    const tok: TokenizerLike = {
      clsId: 2, sepId: 3, maskId: 1, padId: 0, maskToken: "[MASK]",
      encode(s: string): number[] {
        return s.split(/\s+/).filter(Boolean).map((w) => {
          if (!vocab.has(w)) vocab.set(w, 100 + vocab.size);
          return vocab.get(w)!;
        });
      },
    };
    const q: InternalQ = { t: "noul", ins: "Is it urgent?", crit: null };
    const full = buildSequence(tok, "", q, 1e6, 192).ids.length;     // prompt + closing [SEP], no state
    const { ids } = buildSequence(tok, "one two three", q, full + room, 192, undefined, true);
    expect(ids.slice(full - 1)).toEqual([...kept.map((w) => vocab.get(w)!), tok.sepId]);
  });
  it("optionOrder reorders option blocks (py parity)", () => {
    const tok = {
      clsId: 101, sepId: 102, maskId: 103, maskToken: "[MASK]",
      encode(s: string): number[] {
        return s.split(/\s+/).filter(Boolean).map((w) => 2000 + (w.charCodeAt(0) % 500));
      },
    } as any;
    const q = { t: "choice", ins: "pick", crit: { a: "x", b: "y" } } as any;
    const dflt = buildSequence(tok, "hi", q, 64, 32);
    const rev = buildSequence(tok, "hi", q, 64, 32, [1, 0]);
    expect(rev.markers.length).toBe(2);
    expect(rev.ids).not.toEqual(dflt.ids);
    // swapping twice restores identity
    const back = buildSequence(tok, "hi", q, 64, 32, [0, 1]);
    expect(back.ids).toEqual(dflt.ids);
  });
  it("collateItems pads ids/att/mpos/mmask/qtype/label/meta (py parity)", () => {
    const batch = [[
      { ids: [1, 2, 3], markers: [1, 2], qtype: 0 },
      { ids: [4, 5], markers: [1], qtype: 1, label: 2 },
    ]];
    const out = collateItems(batch as any, 0)!;
    expect(out.inputIds).toEqual([[1, 2, 3], [4, 5, 0]]);
    expect(out.attentionMask).toEqual([[1, 1, 1], [1, 1, 0]]);
    expect(out.markerPos).toEqual([[1, 2], [1, 0]]);
    expect(out.markerMask).toEqual([[true, true], [true, false]]);
    expect(out.qtype).toEqual([0, 1]);
    expect(out.label).toEqual([-1, 2]);
    expect(out.meta.length).toBe(2);
    expect(out.meta[0]).toMatchObject({ qtype: 0 });
    expect("ids" in out.meta[0]).toBe(false);
    expect(out.target).toBeUndefined();
  });
  it("collateItems includes target when present and null on empty", () => {
    const batch = [[{ ids: [1], markers: [0, 1], qtype: 0, target: [0.2, 0.8] }]];
    const out = collateItems(batch as any, 0)!;
    expect(out.target).toEqual([[0.2, 0.8]]);
    expect(collateItems([], 0)).toBeNull();
    expect(collateItems([[]], 0)).toBeNull();
  });
  it("collateItems rejects a target longer than its own marker count, even in mixed batches (#311)", () => {
    // the first row's target fits the batch-wide K=4 but exceeds its own 2 markers
    const mixed = [[
      { ids: [1], markers: [0, 1], qtype: 1, target: [0.2, 0.3, 0.5] },
      { ids: [2], markers: [0, 1, 2, 3], qtype: 1, target: [0.25, 0.25, 0.25, 0.25] },
    ]];
    expect(() => collateItems(mixed as any, 0)).toThrow(/one entry per option/);
    const overK = [[{ ids: [1], markers: [0], qtype: 1, target: [0.0, 1.0] }]];
    expect(() => collateItems(overK as any, 0)).toThrow(/one entry per option/);
    // a shorter target is padded with zeros, not rejected
    const short = collateItems([[{ ids: [1], markers: [0, 1, 2], qtype: 1, target: [1.0] }]] as any, 0)!;
    expect(short.target).toEqual([[1.0, 0.0, 0.0]]);
  });
});
