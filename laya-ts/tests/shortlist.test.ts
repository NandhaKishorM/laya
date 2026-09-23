import { describe, expect, it } from "vitest";
import { shortlistChoice } from "../src/shortlist.js";
const emb = async (texts: string[]) =>
  texts.map((t) => (t.includes("refund") || t.includes("billed") ? [1, 0] : [0, 1]));
describe("shortlist", () => {
  it("keeps top-k by cosine", async () => {
    const labels = await shortlistChoice({ text: "billed twice refund" },
      { card_arrival: "where is my card", transfer_fee: "fee on transfer", refund: "refund money" },
      emb as any, 1);
    expect(labels).toEqual(["refund"]);
  });
  it("passthrough when k >= n skips embed", async () => {
    let called = false;
    await shortlistChoice("x", { a: "1" }, (async (t: string[]) => { called = true; return t.map(() => [1]); }) as any, 5);
    expect(called).toBe(false);
  });
});
