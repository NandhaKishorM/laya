import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
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
});
