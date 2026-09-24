// laya-ts/tests/bpe.test.ts — byte-level BPE parity (checkpoint tokenizer is GPT-2-style BPE).
// Fixture source: hand-verified against the checkpoint tokenizer.json
// (transformers NOT installed, so no AutoTokenizer fixture; merges[0..3] ranks used below:
//  rank 0, 1, 2, 3 for the pairs (G,G), (G,t), (G,a), (h,e) where G=U+0120;
//  plus ("a","b")=112, ("t","h")=149 from the same merges list).
import { describe, expect, it } from "vitest";
import { bpeEncode, parseTokenizerJson, CHECKPOINT_IDS } from "../src/tokenizer.js";

const G = String.fromCharCode(0x120); // GPT-2 byte-level space (U+0120)

describe("bpeEncode", () => {
  it("does not lowercase: 'Hello' keeps cap H encoding", () => {
    const vocab = new Map([["H", 1], ["h", 2], ["e", 3], ["l", 4], ["o", 5]]);
    expect(bpeEncode(vocab, new Map(), "Hello")).toEqual([1, 3, 4, 4, 5]);
    expect(bpeEncode(vocab, new Map(), "hello")).toEqual([2, 3, 4, 4, 5]);
  });
  it("NFC-normalizes (composed == decomposed)", () => {
    const vocab = new Map([["c", 1], ["a", 2], ["f", 3], ["Ã", 4], ["©", 5]]);
    expect(bpeEncode(vocab, new Map(), "café")).toEqual(bpeEncode(vocab, new Map(), "café"));
  });
  it("applies merges in rank order (real merges[1] before merges[3])", () => {
    const vocab = new Map([[G, 10], ["t", 20], ["h", 30], ["e", 40], [G + "t", 50], ["he", 60]]);
    const merges = new Map([[G + " t", 1], ["h e", 3]]);
    // " the" -> [G,t,h,e] -> merge (G,t) r1 -> [Gt,h,e] -> merge (h,e) r3 -> [Gt,he]
    expect(bpeEncode(vocab, merges, " the")).toEqual([50, 60]);
  });
  it("lower rank wins between competing pairs", () => {
    const vocab = new Map([["a", 1], ["b", 2], ["c", 3], ["bc", 4], ["ab", 5]]);
    const merges = new Map([["a b", 112], ["b c", 5]]);
    // [a,b,c]: (b,c) r5 beats (a,b) r112 -> [a,bc]
    expect(bpeEncode(vocab, merges, "abc")).toEqual([1, 4]);
  });
  it("'a'+'b' merges per merges list (rank 112)", () => {
    const vocab = new Map([["a", 1], ["b", 2], ["ab", 3]]);
    expect(bpeEncode(vocab, new Map([["a b", 112]]), "ab")).toEqual([3]);
  });
  it("unknown byte -> UNK 50280", () => {
    const vocab = new Map([["[UNK]", 50280], ["a", 1], ["b", 2]]);
    expect(bpeEncode(vocab, new Map(), "axb")).toEqual([1, 50280, 2]);
  });
  it("deterministic on 'hello world', matches hand fixture", () => {
    const vocab = new Map([
      ["h", 1], ["e", 2], ["l", 3], ["o", 4], ["w", 5],
      ["r", 6], ["d", 7], [G, 8], ["he", 9], ["ll", 10],
    ]);
    const merges = new Map([["h e", 0], ["l l", 1]]);
    // "hello" -> [h,e,l,l,o] -> [he,l,l,o] -> [he,ll,o] = [9,10,4]
    // " world" -> [G,w,o,r,l,d], no merges apply = [8,5,4,6,3,7]
    const expected = [9, 10, 4, 8, 5, 4, 6, 3, 7];
    expect(bpeEncode(vocab, merges, "hello world")).toEqual(expected);
    expect(bpeEncode(vocab, merges, "hello world")).toEqual(expected);
  });
});

describe("parseTokenizerJson", () => {
  const hf = {
    model: { vocab: { hello: 1, "[UNK]": 50280 }, merges: [["h", "e"], ["a", "b"]] },
    added_tokens: [
      { id: 50280, content: "[UNK]" },
      { id: 50281, content: "[CLS]" },
      { id: 50282, content: "[SEP]" },
      { id: 50283, content: "[PAD]" },
      { id: 50284, content: "[MASK]" },
    ],
  };
  it("maps specials from added_tokens", () => {
    const data = parseTokenizerJson(hf)!;
    expect(data.ids).toEqual({ cls: 50281, sep: 50282, mask: 50284, pad: 50283, unk: 50280 });
    expect(data.merges.get("h e")).toBe(0);
    expect(data.merges.get("a b")).toBe(1);
  });
  it("checkpoint ids match ground truth", () => {
    expect(CHECKPOINT_IDS).toEqual({ cls: 50281, sep: 50282, mask: 50284, pad: 50283, unk: 50280 });
  });
  it("returns null on garbage", () => {
    expect(parseTokenizerJson(null)).toBeNull();
    expect(parseTokenizerJson({})).toBeNull();
  });
});
