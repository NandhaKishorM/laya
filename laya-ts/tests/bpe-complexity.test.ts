// laya-ts/tests/bpe-complexity.test.ts
//
// Regression: merging one word must stay near-linear in that word's length.
//
// `bpeWord` used to rescan the whole word for its best-ranked pair and rebuild the array
// once per merge -- O(n^2) in the word's length. That is invisible on English, because
// `GPT2_SPLIT` cuts prose into short pieces at every space -- a five-byte median with the
// checkpoint tokenizer, p90 eight. It is not
// invisible on a script that does not space its words: the split takes `\p{L}+`, a maximal
// run of letters, so a whole Chinese or Japanese state arrives as ONE piece. Both encoders
// that call `bpeWord` have this shape -- see the comment on it in src/tokenizer.ts -- and the
// metaspace one is the path CJK actually takes. Measured on node 20 with each checkpoint's own
// tokenizer, 4000 unspaced characters cost 369 ms through `metaspaceEncode` (mmBERT) and 2.7 s
// through `bpeEncode` (ModernBERT), against ~1 ms for English prose of the same length, growing
// 4.0x per doubling either way. Both encoders are synchronous, so that time is the whole event
// loop, not one slow request.
//
// Parity below is asserted against the previous single-path algorithm, kept in this file,
// rather than against hand-written token ids -- the point is that nothing moved.
//
// Fixtures are built here, not downloaded: the checkpoint tokenizer.json is not in the repo
// and transformers is not installed (same constraint as bpe.test.ts).
import { describe, expect, it } from "vitest";
import { bpeEncode, metaspaceEncode, METASPACE_REPLACEMENT } from "../src/tokenizer.js";

// ---------------------------------------------------------------- fixtures ----
/** GPT-2 byte<->unicode table. A copy of the one in src/tokenizer.ts; `refEncode` needs it
 *  to reproduce the whole pipeline, and the harness self-check below proves the copy is
 *  faithful. */
function byteToUnicode(): Map<number, string> {
  const b2u = new Map<number, string>();
  const ranges: Array<[number, number]> = [[0x21, 0x7e], [0xa1, 0xac], [0xae, 0xff]];
  const inRange = (b: number): boolean => ranges.some(([lo, hi]) => b >= lo && b <= hi);
  let k = 0;
  for (let b = 0; b < 256; b++) {
    const cp = inRange(b) ? b : (k++ < 0x100 ? k - 1 + 0x100 : k - 1);
    b2u.set(b, String.fromCodePoint(cp));
  }
  return b2u;
}
const B2U = byteToUnicode();
const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const ENC = new TextEncoder();
const byteChars = (piece: string): string[] => {
  const out: string[] = [];
  for (const b of ENC.encode(piece)) out.push(B2U.get(b) ?? "");
  return out;
};

/** The algorithm as it shipped before the heap: rescan for the best pair, once per merge. */
function bpeWordPrevious(chars: string[], rank: Map<string, number>): string[] {
  let word = chars.slice();
  if (word.length <= 1) return word;
  for (;;) {
    let best = Infinity, idx = -1;
    for (let i = 0; i < word.length - 1; i++) {
      const r = rank.get(word[i] + " " + word[i + 1]);
      if (r !== undefined && r < best) { best = r; idx = i; }
    }
    if (idx < 0) return word;
    word = [...word.slice(0, idx), word[idx] + word[idx + 1], ...word.slice(idx + 2)];
  }
}

/** Reference encode: the shipped pipeline with the previous merge function spliced in. */
function refEncode(vocab: Map<string, number>, merges: Map<string, number>, text: string): number[] {
  const unk = vocab.get("[UNK]") ?? 50280;
  const out: number[] = [];
  for (const piece of text.normalize("NFC").match(SPLIT) ?? []) {
    for (const tok of bpeWordPrevious(byteChars(piece), merges)) out.push(vocab.get(tok) ?? unk);
  }
  return out;
}

/** The Metaspace pipeline with the previous merge function spliced in. Mirrors
 *  `metaspaceEncode` for the default ' ' -> marker replacement. */
function refMetaspaceEncode(
  vocab: Map<string, number>, merges: Map<string, number>, text: string, unkId?: number,
): number[] {
  const unk = unkId ?? vocab.get("<unk>") ?? vocab.get("[UNK]") ?? 50280;
  if (!text) return [];
  const t = text.split(" ").join(METASPACE_REPLACEMENT);
  const out: number[] = [];
  const push = (piece: string): void => {
    for (const tok of bpeWordPrevious(Array.from(piece), merges)) out.push(vocab.get(tok) ?? unk);
  };
  for (const seg of t.split(/(\n+)/)) {
    if (!seg) continue;
    if (seg[0] === "\n") { push(seg); continue; }
    const w = seg.startsWith(METASPACE_REPLACEMENT) ? seg : METASPACE_REPLACEMENT + seg;
    for (const chunk of w.split(METASPACE_REPLACEMENT).slice(1)) {
      push(chunk ? METASPACE_REPLACEMENT + chunk : METASPACE_REPLACEMENT);
    }
  }
  return out;
}

/** Metaspace fixture: merges over unicode CHARACTERS (no byte map), trained the same way. */
function trainMetaspace(corpus: string[], rounds: number): { vocab: Map<string, number>; merges: Map<string, number> } {
  const words: string[][] = [];
  for (const text of corpus) {
    const t = text.split(" ").join(METASPACE_REPLACEMENT);
    const w = t.startsWith(METASPACE_REPLACEMENT) ? t : METASPACE_REPLACEMENT + t;
    for (const chunk of w.split(METASPACE_REPLACEMENT).slice(1)) {
      words.push(Array.from(chunk ? METASPACE_REPLACEMENT + chunk : METASPACE_REPLACEMENT));
    }
  }
  const merges = new Map<string, number>();
  const vocab = new Map<string, number>([["<unk>", 3]]);
  let nextId = 10;
  for (const w of words) for (const c of w) if (!vocab.has(c)) vocab.set(c, nextId++);
  for (let r = 0; r < rounds; r++) {
    const freq = new Map<string, number>();
    for (const w of words) {
      for (let i = 0; i < w.length - 1; i++) {
        const k = w[i] + " " + w[i + 1];
        freq.set(k, (freq.get(k) ?? 0) + 1);
      }
    }
    let bestKey: string | null = null, bestN = 0;
    for (const [k, v] of freq) if (v > bestN || (v === bestN && bestKey !== null && k < bestKey)) { bestN = v; bestKey = k; }
    if (bestKey === null || bestN < 2) break;
    merges.set(bestKey, r);
    const [a, b] = bestKey.split(" ") as [string, string];
    const joined = a + b;
    if (!vocab.has(joined)) vocab.set(joined, nextId++);
    for (let wi = 0; wi < words.length; wi++) {
      const w = words[wi], out: string[] = [];
      for (let i = 0; i < w.length;) {
        if (i < w.length - 1 && w[i] === a && w[i + 1] === b) { out.push(joined); i += 2; } else { out.push(w[i]); i += 1; }
      }
      words[wi] = out;
    }
  }
  return { vocab, merges };
}

/** Train a small BPE table over a corpus: the most frequent adjacent pair each round, so
 *  merges are multi-level and dense the way a learned table is, not one flat pass. */
function trainMerges(corpus: string[], rounds: number): { vocab: Map<string, number>; merges: Map<string, number> } {
  const words: string[][] = [];
  for (const text of corpus) {
    for (const piece of text.normalize("NFC").match(SPLIT) ?? []) words.push(byteChars(piece));
  }
  const merges = new Map<string, number>();
  const vocab = new Map<string, number>([["[UNK]", 50280]]);
  let nextId = 1;
  for (const w of words) for (const c of w) if (!vocab.has(c)) vocab.set(c, nextId++);
  for (let r = 0; r < rounds; r++) {
    const freq = new Map<string, number>();
    for (const w of words) {
      for (let i = 0; i < w.length - 1; i++) {
        const k = w[i] + " " + w[i + 1];
        freq.set(k, (freq.get(k) ?? 0) + 1);
      }
    }
    let bestKey: string | null = null, bestN = 0;
    for (const [k, v] of freq) if (v > bestN || (v === bestN && bestKey !== null && k < bestKey)) { bestN = v; bestKey = k; }
    if (bestKey === null || bestN < 2) break;
    merges.set(bestKey, r);
    const [a, b] = bestKey.split(" ") as [string, string];
    const joined = a + b;
    if (!vocab.has(joined)) vocab.set(joined, nextId++);
    for (let wi = 0; wi < words.length; wi++) {
      const w = words[wi], out: string[] = [];
      for (let i = 0; i < w.length;) {
        if (i < w.length - 1 && w[i] === a && w[i + 1] === b) { out.push(joined); i += 2; } else { out.push(w[i]); i += 1; }
      }
      words[wi] = out;
    }
  }
  return { vocab, merges };
}

const ZH = "客户被重复扣款了两次现在想要退款请尽快处理这个问题谢谢您的帮助";
const JA = "お客様が二重に請求されたため返金を希望しています至急対応してください";
const EN = "The customer was billed twice and wants a refund please help me resolve this today";
const { vocab, merges } = trainMerges([EN, ZH, JA, "abcdefghij".repeat(4), "aaaaaaaaaaaaaaaa"], 300);

/** A doubling table over one character: ("a","a") -> ("aa","aa") -> ("aaaa","aaaa") ...,
 *  each rank lower than the last. A run of n characters collapses to exactly ONE token of
 *  length n in n-1 merges, which is the most merges a word of length n can take and so the
 *  cleanest quadratic signal available -- and the table itself is only log2(n) entries.
 *
 *  A "chain" table (("a","a"), ("aa","a"), ("aaa","a"), ...) looks equivalent and is not: the
 *  lowest rank is always ("a","a"), so greedy starts a fresh pair rather than extending and a
 *  run collapses to n/2 two-character tokens, never one long token. */
function doublingTable(ch: string, maxPow: number): { vocab: Map<string, number>; merges: Map<string, number> } {
  const merges = new Map<string, number>();
  const vocab = new Map<string, number>([["[UNK]", 50280], [ch, 1]]);
  let id = 2;
  for (let k = 0; k <= maxPow; k++) {
    const half = ch.repeat(2 ** k);
    merges.set(half + " " + half, k);
    vocab.set(half + half, id++);
  }
  return { vocab, merges };
}
const oneToken = doublingTable("a", 16);

const repeatTo = (unit: string, n: number): string =>
  unit.repeat(Math.ceil(n / unit.length)).slice(0, n);
/** Warm up once, then take the best of five: JIT tiering and a descheduled run on a shared
 *  runner both inflate a single sample. (The lesson from commit 970dc8c on #383.) */
function bestMs(fn: () => void, reps = 5): number {
  fn();
  let best = Infinity;
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    fn();
    best = Math.min(best, performance.now() - t0);
  }
  return best;
}

// ------------------------------------------------------------------ parity ----
describe("bpe merging: parity with the previous algorithm", () => {
  it("the reference harness reproduces the shipped pipeline on short input", () => {
    // Short pieces exercise the least of the merge machinery, so a disagreement here is the
    // harness's own byte map or split rather than the merge order. Passing is what licenses
    // refEncode as the oracle for the long cases below.
    for (const text of [EN, "hello world", "café", "a.b@c", "9.9 abc"]) {
      expect(bpeEncode(vocab, merges, text)).toEqual(refEncode(vocab, merges, text));
    }
  });

  const cases: Array<[string, string]> = [
    ["english", EN],
    ["chinese", ZH],
    ["japanese", JA],
    ["chinese x8 (one long piece)", ZH.repeat(8)],
    ["japanese x8 (one long piece)", JA.repeat(8)],
    ["mixed scripts and punctuation", `${EN} ${ZH} github.com user@acme.com v1.2.3 $42.50 🙏`],
    ["one 600-char token", "a".repeat(600)],
    ["31-character run", "a".repeat(31)],
    ["32-character run", "a".repeat(32)],
    ["33-character run", "a".repeat(33)],
    ["empty", ""],
    ["single char", "a"],
    ["whitespace only", "   \n\t "],
    ["digits and symbols", "1234567890 !!!??? ---___"],
  ];
  for (const [label, text] of cases) {
    it(`encodes ${label} identically`, () => {
      expect(bpeEncode(vocab, merges, text)).toEqual(refEncode(vocab, merges, text));
    });
  }

  it("agrees with the previous algorithm on the doubling table at every length 0..200", () => {
    // This table merges at every step, so every length in the range does real work rather
    // than returning early -- an exhaustive sweep of the short and middle sizes, where a
    // linked-list or heap off-by-one would show up first.
    for (let n = 0; n <= 200; n++) {
      const s = "a".repeat(n);
      expect(bpeEncode(oneToken.vocab, oneToken.merges, s)).toEqual(refEncode(oneToken.vocab, oneToken.merges, s));
    }
  });

  it("agrees with the previous algorithm on 20 000 randomised strings", () => {
    let seed = 12345;
    // Math.imul, not `*`: the product of two 32-bit values exceeds 2^53, so a plain multiply
    // rounds the low bits away and the generator collapses -- this same line written with `*`
    // yields 1 676 distinct strings out of 20 000 iterations here instead of 18 706.
    const rnd = (): number => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
    const alphabets = ["ab ", "abcdefghij", "客户退款お客様", "aA1 .@-_客户é🙏", "aaaaaaab"];
    let mismatch: string | null = null;
    for (let i = 0; i < 20_000 && mismatch === null; i++) {
      const alphabet = alphabets[i % alphabets.length];
      const chars = Array.from(alphabet);
      let s = "";
      const len = Math.floor(rnd() * 70);
      for (let j = 0; j < len; j++) s += chars[Math.floor(rnd() * chars.length)];
      const got = bpeEncode(vocab, merges, s), want = refEncode(vocab, merges, s);
      if (got.length !== want.length || got.some((v, k) => v !== want[k])) mismatch = s;
    }
    expect(mismatch).toBeNull();
  });
});

// ------------------------------------------------- metaspace (multilingual) ----
// The checkpoint that actually receives CJK is `multilingual` (mmBERT-base, see
// laya/router.py), whose tokenizer is Gemma/SentencePiece-style -- so it runs
// `metaspaceEncode`, not `bpeEncode`. That path shares `bpeWord`, and it cuts pieces only at
// the metaspace marker (i.e. at spaces) and at newline runs, so an unspaced state is one piece
// there too. It merges unicode CHARACTERS rather than bytes, so the same text is a shorter
// word than on the byte-level path -- a smaller n on the same quadratic curve.
const ms = trainMetaspace([EN, ZH, JA, "abcdefghij".repeat(4)], 300);
const msOneToken = doublingTable("客", 16);

describe("metaspace merging: parity and complexity", () => {
  it("the reference harness reproduces the shipped metaspace pipeline on short input", () => {
    for (const text of [EN, "hello world", "a b c", ""]) {
      expect(metaspaceEncode(ms.vocab, ms.merges, text)).toEqual(refMetaspaceEncode(ms.vocab, ms.merges, text));
    }
  });

  for (const [label, text] of [
    ["chinese", ZH], ["japanese", JA], ["chinese x8 (one piece)", ZH.repeat(8)],
    ["english", EN], ["newline runs", `${EN}\n\n${ZH}\n${JA}`],
    ["leading and trailing spaces", `  ${EN}  `], ["empty", ""], ["single space", " "],
  ] as Array<[string, string]>) {
    it(`encodes ${label} identically on the metaspace path`, () => {
      expect(metaspaceEncode(ms.vocab, ms.merges, text)).toEqual(refMetaspaceEncode(ms.vocab, ms.merges, text));
    });
  }

  it("agrees with the previous algorithm at every length 0..200", () => {
    for (let n = 0; n <= 200; n++) {
      const s = "客".repeat(n);
      expect(metaspaceEncode(msOneToken.vocab, msOneToken.merges, s))
        .toEqual(refMetaspaceEncode(msOneToken.vocab, msOneToken.merges, s));
    }
  });

  it("an unspaced state is bounded on the metaspace path", () => {
    // Previous algorithm / ceiling / current for THIS fixture, measured on node 20 via
    // metaspaceEncode:  4 000 chars  389 ms / 25 ms / 3.4 ms   (16x / 7x)
    // With mmBERT's real tokenizer the same input is 369 ms before and 0.77 ms after; the
    // fixture is slower after only because its merge table is denser per character.
    // Cheaper than the byte-level path at the same character count, because this encoder
    // merges unicode characters and CJK is three bytes to the character -- the same curve at
    // a third of the n. The revert ratio is 16.2, so the shape is unchanged.
    expect(bestMs(() => metaspaceEncode(msOneToken.vocab, msOneToken.merges, "客".repeat(4_000))))
      .toBeLessThan(25);
  });

  it("4x the input costs under 10x the time on the metaspace path", () => {
    const at = (n: number): number =>
      bestMs(() => metaspaceEncode(msOneToken.vocab, msOneToken.merges, "客".repeat(n)));
    const small = at(4_000);
    const large = at(16_000);
    if (small > 0.05) expect(large / small).toBeLessThan(10);
  });
});

// -------------------------------------------------------------- complexity ----
describe("bpe merging: complexity", () => {
  // These go through the exported `bpeEncode`, so reverting src/tokenizer.ts fails them.
  // Testing a copy of the algorithm in this file would only ever test the copy.
  //
  // Each ceiling sits near the geometric mean of the two costs it has to separate, so there
  // is room both for a slow runner and for catching a revert on a fast one. Measured on
  // node 20 as previous cost / ceiling / current cost:
  //   4 000 chars       249 ms /  25 ms / 2.1 ms    (10x / 12x)
  //  16 000 chars     4 012 ms / 180 ms / 8.4 ms    (22x / 21x)
  //   8 000 x one char 1 039 ms /  60 ms / 3.9 ms    (17x / 15x)
  it("4 000 characters collapsing to one token is bounded", () => {
    expect(bestMs(() => bpeEncode(oneToken.vocab, oneToken.merges, "a".repeat(4_000)))).toBeLessThan(25);
  });

  it("16 000 characters collapsing to one token is bounded", () => {
    expect(bestMs(() => bpeEncode(oneToken.vocab, oneToken.merges, "a".repeat(16_000)))).toBeLessThan(180);
  });

  it("an unspaced CJK state costs about what spaced prose of the same size costs", () => {
    // The real shape of the bug: same byte count, same tokenizer, only the spacing differs.
    // A ratio is machine-independent in a way a wall-clock ceiling is not -- it was ~2000x
    // before and is single digits now.
    const zh = repeatTo(ZH, 4_000);
    const en = repeatTo(`${EN} `, 4_000);
    const ratio = bestMs(() => bpeEncode(vocab, merges, zh)) /
      Math.max(bestMs(() => bpeEncode(vocab, merges, en)), 0.05);
    expect(ratio).toBeLessThan(25);
  });

  it("4x the input costs under 10x the time (n log n ~4.6, quadratic ~16)", () => {
    const at = (n: number): number => bestMs(() => bpeEncode(oneToken.vocab, oneToken.merges, "a".repeat(n)));
    const small = at(4_000);
    const large = at(16_000);
    // The ceiling is 10, not 8. The fix is O(n log n), not linear, so 4x the input predicts
    // 4 * log(16000)/log(4000) ~= 4.6 rather than 4.0 -- measured 4.71 here, against 16.05
    // on a revert. 10 sits between the two with comparable margin either side; 8 would leave
    // only 1.8x of headroom above a legitimate result and flake on a loaded runner.
    // performance.now() granularity makes a sub-0.05 ms baseline meaningless; the absolute
    // ceilings above already carry the guarantee in that case.
    if (small > 0.05) expect(large / small).toBeLessThan(10);
  });

  it("a long run does not depend on which character it is", () => {
    // Each character gets its own doubling table, so merging actually applies to all three.
    // Reusing the trained table here would be vacuous: it holds ("a","a") but no ("z","z") or
    // ("1","1"), so those runs take no merge at all and pass on the unfixed code too.
    for (const ch of ["a", "z", "1"]) {
      const t = doublingTable(ch, 14);
      expect(bestMs(() => bpeEncode(t.vocab, t.merges, ch.repeat(8_000)))).toBeLessThan(60);
    }
  });
});
