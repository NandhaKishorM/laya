// laya-ts/tests/parity-sequence.test.ts
import { describe, expect, it } from "vitest";
import { renderOptions, serializeState, buildSequence, collateItems } from "../src/common.js";
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
});
