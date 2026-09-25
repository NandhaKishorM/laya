import { describe, expect, it } from "vitest";
import { buildQuestionPrefix } from "../src/common.js";
import { defaultTokenizer } from "../src/agent.js";

describe("prefix cache", () => {
  it("returns cached prefix on repeat", () => {
    const tok = defaultTokenizer();
    const q: any = { t: "choice", ins: "What?", crit: { a: "x", b: "y" } };
    const p1 = buildQuestionPrefix(tok, q, 512, 192);
    const p2 = buildQuestionPrefix(tok, q, 512, 192);
    expect(p2).toBe(p1);
    expect(p2.ids).toEqual(p1.ids);
  });
  it("does not merge undefined descriptions into missing keys", () => {
    const tok = defaultTokenizer();
    const withUndef: any = { t: "choice", ins: "What?", crit: { a: undefined, b: "x" } };
    const without: any = { t: "choice", ins: "What?", crit: { b: "x" } };
    const p1 = buildQuestionPrefix(tok, withUndef, 512, 192);
    const p2 = buildQuestionPrefix(tok, without, 512, 192);
    expect(p2).not.toBe(p1);
    expect(p1.markers.length).toBe(2);
    expect(p2.markers.length).toBe(1);
  });
  it("tolerates BigInt descriptions", () => {
    const tok = defaultTokenizer();
    const q: any = { t: "choice", ins: "What?", crit: { a: 1n } };
    expect(() => buildQuestionPrefix(tok, q, 512, 192)).not.toThrow();
  });
});
