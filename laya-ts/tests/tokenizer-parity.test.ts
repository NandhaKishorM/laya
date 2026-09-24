// laya-ts/tests/tokenizer-parity.test.ts — Gemma-style (Metaspace BPE) parity.
// Reference vectors from transformers AutoTokenizer on model-ml/tokenizer.json,
// add_special_tokens=False. Hardcoded so the suite passes on any machine; when the
// real model-ml/tokenizer.json is present it is loaded and checked byte-identical.
import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { encodeWithData, metaspaceEncode, parseTokenizerJson } from "../src/tokenizer.js";

const EN = "choice question: Which department should handle this request?";
const EN_IDS = [6241, 2872, 235292, 12236, 9888, 1412, 6589, 736, 3853, 235336];
const HI = "मुझसे दो बार शुल्क लिया गया";
const HI_IDS = [39728, 238385, 20579, 51484, 99744, 144905, 228834, 158850, 44370];

const here = dirname(fileURLToPath(import.meta.url));
const mlPath = resolve(here, "../../model-ml/tokenizer.json");

describe("metaspaceEncode (synthetic)", () => {
  const vocab = new Map([
    ["▁hi", 10], ["▁", 11], ["h", 12], ["i", 13], ["<unk>", 3],
  ]);
  const joins = new Map([["▁ h", 0], ["▁h i", 1]]);
  it("prepends one marker, splits keeping marks, no lowercase", () => {
    expect(metaspaceEncode(vocab, joins, "hi")).toEqual([10]);
    expect(metaspaceEncode(vocab, joins, " hi")).toEqual([10]);
    expect(metaspaceEncode(vocab, joins, "  hi")).toEqual([11, 10]);
    expect(metaspaceEncode(vocab, new Map(), "")).toEqual([]);
    expect(metaspaceEncode(vocab, joins, "HI")).toEqual([11, 3, 3]);
  });
  it("unknown piece -> unk id", () => {
    expect(metaspaceEncode(vocab, joins, "z")).toEqual([11, 3]);
  });
});

describe("parseTokenizerJson specials aliases", () => {
  it("resolves Gemma <bos>/<eos>/<pad>/<mask>/<unk> (cls=2 sep=1 pad=0 mask=4 unk=3)", () => {
    const data = parseTokenizerJson({
      model: { vocab: { "▁a": 10 }, merges: [] },
      normalizer: { type: "Replace", pattern: { String: " " }, content: "▁" },
      pre_tokenizer: { type: "Metaspace", replacement: "▁", prepend_scheme: "always", split: true },
      added_tokens: [
        { id: 0, content: "<pad>" }, { id: 1, content: "<eos>" }, { id: 2, content: "<bos>" },
        { id: 3, content: "<unk>" }, { id: 4, content: "<mask>" },
      ],
    })!;
    expect(data.ids).toEqual({ cls: 2, sep: 1, mask: 4, pad: 0, unk: 3 });
    expect(data.kind).toBe("metaspace");
    expect(data.maskToken).toBe("<mask>");
    expect(data.replaces).toEqual([[" ", "▁"]]);
  });
  it("keeps ModernBERT [CLS]/[SEP]/[PAD]/[MASK]/[UNK] on the ByteLevel path", () => {
    const data = parseTokenizerJson({
      model: { vocab: { hello: 1 }, merges: ["h e"] },
      pre_tokenizer: { type: "ByteLevel" },
      added_tokens: [
        { id: 50280, content: "[UNK]" }, { id: 50281, content: "[CLS]" },
        { id: 50282, content: "[SEP]" }, { id: 50283, content: "[PAD]" },
        { id: 50284, content: "[MASK]" },
      ],
    })!;
    expect(data.ids).toEqual({ cls: 50281, sep: 50282, mask: 50284, pad: 50283, unk: 50280 });
    expect(data.kind).toBe("bytelevel");
    expect(data.maskToken).toBe("[MASK]");
  });
});

describe("model-ml/tokenizer.json parity (gated on file presence)", () => {
  it("reproduces reference id-sequences byte-identical", () => {
    if (!existsSync(mlPath)) return;
    const raw = JSON.parse(readFileSync(mlPath, "utf8"));
    const data = parseTokenizerJson(raw)!;
    expect(data.kind).toBe("metaspace");
    expect(data.ids).toMatchObject({ cls: 2, sep: 1, mask: 4, pad: 0, unk: 3 });
    expect(data.maskToken).toBe("<mask>");
    expect(encodeWithData(data, EN)).toEqual(EN_IDS);
    expect(encodeWithData(data, HI)).toEqual(HI_IDS);
  });
});
