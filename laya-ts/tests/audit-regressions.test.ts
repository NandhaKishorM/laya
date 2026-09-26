import { describe, expect, it } from "vitest";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, checkQuestion, toInternal } from "../src/agent.js";
import { buildSequence } from "../src/common.js";
import { loadNodeBundle } from "../src/providers.js";

const tokenizer = () => ({
  clsId: 1,
  sepId: 2,
  maskId: 3,
  padId: 0,
  maskToken: "[MASK]",
  encode(text: string): number[] {
    return Array.from(text).map((c) => c.codePointAt(0) ?? 0);
  },
});

const provider = (calls: any[] = []) => ({
  async runEncoder(batch: any) {
    calls.push(batch);
    return { lastHidden: [[[0, 0]]] };
  },
  async runHead(_hidden: any, _batch: any) {
    return { logits: [[1, 0]], act: [[1, 0]] };
  },
});

describe("audit regressions", () => {
  it("preserves the newest conversation turn", () => {
    const q = toInternal({ type: "noul", instructions: "?" });
    const result = buildSequence(tokenizer(), [
      { text: "old context that must be truncated" },
      { text: "newest refund request" },
    ], q, 64, 32, undefined, true);
    const rendered = result.ids.map((id) => String.fromCodePoint(id)).join("");
    expect(rendered).toContain("refund request");
  });

  it("renders custom noul labels and validates unsupported labels", () => {
    const q = toInternal({
      type: "noul",
      instructions: "?",
      criteria: {},
      labels: { false: "No", true: "Yes" },
    });
    const sequence = buildSequence(tokenizer(), "state", q, 64, 32);
    const rendered = sequence.ids.map((id) => String.fromCodePoint(id)).join("");
    expect(rendered).toContain("No");
    expect(rendered).toContain("Yes");
    expect(() => checkQuestion("q", { type: "noul", instructions: "?", labels: { false: "No" } })).toThrow("labels");
    expect(() => checkQuestion("q", { type: "choice", instructions: "?", criteria: ["yes"], labels: { false: "No", true: "Yes" } })).toThrow("labels");
  });

  it("rejects a null score level, as Python does (#302)", () => {
    expect(() => checkQuestion("q", { type: "score", instructions: "?", criteria: ["low", null, "high"] })).toThrow("level 1");
    expect(() => checkQuestion("q", { type: "score", instructions: "?", criteria: ["low", "mid", undefined] })).toThrow("level 2");
  });

  it("rejects malformed provider output", async () => {
    const agent = new Agent({ provider: {
      async runEncoder(_batch: any) { return { lastHidden: [[[0, 0]]] }; },
      async runHead(_hidden: any, _batch: any) { return { logits: [[]], act: [[1, 0]] }; },
    }, tok: tokenizer() } as any);
    await expect(agent.systemOne("state", { q: { type: "noul", instructions: "?" } })).rejects.toThrow("invalid output");
  });

  it("loads the nested Hugging Face tokenizer layout", async () => {
    const root = await mkdtemp(join(tmpdir(), "laya-ts-audit-"));
    try {
      await mkdir(join(root, "tokenizer"));
      await writeFile(join(root, "rl_agent_config.json"), JSON.stringify({ max_len: 32 }));
      await writeFile(join(root, "tokenizer", "tokenizer.json"), JSON.stringify({ model: { vocab: {}, merges: [] } }));
      const bundle = await loadNodeBundle(root);
      expect(bundle.tokenizerJson).toEqual({ model: { vocab: {}, merges: [] } });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
