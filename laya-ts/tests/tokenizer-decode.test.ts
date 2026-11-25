import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { encodeWithData, parseTokenizerJson } from "../src/tokenizer.js";

const fixture = JSON.parse(readFileSync(new URL("../fixtures/tokenizer-decode.json", import.meta.url), "utf8"));

describe("checkpoint decode/re-encode parity", () => {
  for (const [kind, raw] of Object.entries(fixture.tokenizers)) {
    const section = raw as {
      tokenizer: unknown;
      cases: Array<{ name: string; ids: number[]; decoded: string; reencoded: number[] }>;
    };
    const data = parseTokenizerJson(section.tokenizer)!;
    for (const vector of section.cases) {
      it(`${kind}: ${vector.name}`, () => {
        const decoded = data.decode!(vector.ids);
        expect(decoded).toBe(vector.decoded);
        expect(encodeWithData(data, decoded)).toEqual(vector.reencoded);
      });
    }
  }

  it("records actual multilingual re-encoding growth instead of promising lossless windows", () => {
    const vector = fixture.tokenizers.multilingual.cases.find((c: { name: string }) => c.name === "reencoding-growth");
    expect(vector.ids).toHaveLength(33);
    expect(vector.reencoded).toHaveLength(34);
  });
});

describe("decoder boundaries", () => {
  const byteLevel = (vocab: Record<string, number>, added_tokens: unknown[] = []) => parseTokenizerJson({
    model: { vocab, merges: [] }, decoder: { type: "ByteLevel" }, added_tokens,
  })!;

  it("decodes UTF-8 across token boundaries and preserves BOM/NUL", () => {
    const data = byteLevel({ "ð": 0, "Ł": 1, "ĺ": 2, "Ģ": 3 });
    expect(data.decode!([0, 1, 2, 3])).toBe("😀");
    const bom = byteLevel({ "ï": 0, "»": 1, "¿": 2, "Ā": 3 });
    expect(bom.decode!([0, 1, 2, 3])).toBe("\ufeff\0");
  });

  it("falls back to an unmappable added token as a whole, retaining special tokens", () => {
    const data = byteLevel({ "Ġ": 0 }, [
      { id: 1, content: "Ġ🫠", special: true },
      { id: 2, content: "[MASK]", special: true },
    ]);
    expect(data.decode!([0, 1, 2, 999])).toBe(" Ġ🫠[MASK]");
  });

  it("keeps encoding available for absent or unsupported decoders", () => {
    for (const decoder of [undefined, { type: "Metaspace" }, { type: "Sequence", decoders: [
      { type: "Replace", pattern: { Regex: "▁" }, content: " " },
      { type: "ByteFallback" }, { type: "Fuse" },
    ] }, { type: "Sequence", decoders: [
      { type: "Replace", pattern: { String: "▁" }, content: " " },
      { type: "ByteFallback" }, { type: "Fuse" }, { type: "Strip" },
    ] }]) {
      const data = parseTokenizerJson({ model: { vocab: { a: 1 }, merges: [] }, decoder })!;
      expect(data.decode).toBeUndefined();
      expect(encodeWithData(data, "a")).toEqual([1]);
    }
  });

  it("loads a decoder into Agent without needing a Node tokenizer dependency", async () => {
    const providers = await import("../src/providers.js");
    const { Agent } = await import("../src/agent.js");
    const bundle = vi.spyOn(providers, "loadWebBundle").mockResolvedValue({
      cfg: {}, tokenizerJson: fixture.tokenizers.english.tokenizer,
      dir: "https://example.test/model", revision: null,
    } as any);
    const provider = vi.spyOn(providers, "createWebProvider").mockResolvedValue({} as any);
    vi.stubGlobal("window", {});
    try {
      const agent = await Agent.load("https://example.test/model");
      const vector = fixture.tokenizers.english.cases.find((c: { name: string }) => c.name === "special");
      expect(agent.tok.decode!(vector.ids)).toBe(vector.decoded);
    } finally { bundle.mockRestore(); provider.mockRestore(); vi.unstubAllGlobals(); }
  });
});
