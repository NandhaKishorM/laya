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
});
