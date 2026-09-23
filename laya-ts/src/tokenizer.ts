export interface TokenizerLike {
  readonly clsId: number; readonly sepId: number;
  readonly maskId: number; readonly padId: number;
  readonly maskToken: string;
  encode(text: string): number[];
}

/** Special ids of the laya ModernBERT checkpoint (HF added_tokens). */
export const CHECKPOINT_IDS = { cls: 50281, sep: 50282, mask: 50284, pad: 50283, unk: 50280 } as const;

/** Alias lookup order per special: ModernBERT `[X]` names first, Gemma `<x>` names after. */
export const SPECIAL_ALIASES = {
  cls: ["[CLS]", "<bos>", "<s>"],
  sep: ["[SEP]", "<eos>", "</s>"],
  pad: ["[PAD]", "<pad>"],
  mask: ["[MASK]", "<mask>"],
  unk: ["[UNK]", "<unk>"],
} as const;

export interface TokenizerIds { cls: number; sep: number; mask: number; pad: number; unk: number }
export type PreTokenizerKind = "metaspace" | "bytelevel";
export interface TokenizerData {
  vocab: Map<string, number>;
  merges: Map<string, number>;
  ids: TokenizerIds;
  kind: PreTokenizerKind;
  maskToken: string;
  /** Normalizer Replace rules (pattern -> content) applied in order before pre-tokenizing. */
  replaces: Array<[string, string]>;
}

/** Metaspace word-boundary marker (HF SentencePiece-style replacement for ' '). */
export const METASPACE_REPLACEMENT = "▁";

/** GPT-2 byte<->unicode table (same mapping as HF ByteLevel pre-tokenizer). */
function byteUnicodeMaps(): { b2u: Map<number, string>; u2b: Map<string, number> } {
  const b2u = new Map<number, string>();
  const u2b = new Map<string, number>();
  const extra = (n: number): number => (n < 0x100 ? n + 0x100 : n);
  const ranges: Array<[number, number]> = [[0x21, 0x7e], [0xa1, 0xac], [0xae, 0xff]];
  let k = 0;
  const inRange = (b: number): boolean => ranges.some(([lo, hi]) => b >= lo && b <= hi);
  for (let b = 0; b < 256; b++) {
    const cp = inRange(b) ? b : extra(k++);
    b2u.set(b, String.fromCodePoint(cp));
    u2b.set(String.fromCodePoint(cp), b);
  }
  return { b2u, u2b };
}

let cached: { b2u: Map<number, string>; u2b: Map<string, number> } | null = null;
function maps(): { b2u: Map<number, string>; u2b: Map<string, number> } {
  if (!cached) cached = byteUnicodeMaps();
  return cached;
}

/** GPT-2 pre-tokenizer split (HF ByteLevel use_regex=true). */
const GPT2_SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** Merge one word's chars greedily by merge rank (first occurrence per step). */
function bpeWord(chars: string[], rank: Map<string, number>): string[] {
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

/** Byte-level BPE encode: NFC-normalize, NO lowercasing, GPT-2 byte map + rank-order merges. */
export function bpeEncode(vocab: Map<string, number>, merges: Map<string, number>, text: string): number[] {
  const { b2u } = maps();
  const unkId = vocab.get("[UNK]") ?? CHECKPOINT_IDS.unk;
  const out: number[] = [];
  const enc = new TextEncoder();
  const parts = text.normalize("NFC").match(GPT2_SPLIT);
  if (!parts) return out;
  for (const piece of parts) {
    const chars: string[] = [];
    for (const b of enc.encode(piece)) chars.push(b2u.get(b) ?? "");
    for (const tok of bpeWord(chars, merges)) out.push(vocab.get(tok) ?? unkId);
  }
  return out;
}

/** Metaspace (SentencePiece-style) BPE encode: unicode chars, NO byte map, NO lowercasing.
 * Normalizer replaces run first, then one marker is ensured at text start, then the text
 * is cut into words at each marker (marker kept as word prefix) with maximal `\n` runs
 * as their own pieces. Each piece is BPE-merged; leftover unknown pieces map to unk. */
export function metaspaceEncode(
  vocab: Map<string, number>,
  merges: Map<string, number>,
  text: string,
  unkId?: number,
  replaces: ReadonlyArray<readonly [string, string]> = [[" ", METASPACE_REPLACEMENT]],
): number[] {
  const unk = unkId ?? vocab.get("<unk>") ?? vocab.get("[UNK]") ?? CHECKPOINT_IDS.unk;
  if (!text) return [];
  let t = text;
  for (const [from, to] of replaces) t = t.split(from).join(to);
  const out: number[] = [];
  const push = (piece: string): void => {
    for (const tok of bpeWord(Array.from(piece), merges)) out.push(vocab.get(tok) ?? unk);
  };
  for (const seg of t.split(/(\n+)/)) {
    if (!seg) continue;
    if (seg[0] === "\n") {
      push(seg);
    } else {
      const w = seg.startsWith(METASPACE_REPLACEMENT) ? seg : METASPACE_REPLACEMENT + seg;
      for (const chunk of w.split(METASPACE_REPLACEMENT).slice(1)) {
        push(chunk ? METASPACE_REPLACEMENT + chunk : METASPACE_REPLACEMENT);
      }
    }
  }
  return out;
}

/** Dispatch to the Metaspace or GPT-2/ByteLevel encoder based on the parsed pre-tokenizer. */
export function encodeWithData(data: TokenizerData, text: string): number[] {
  return data.kind === "metaspace"
    ? metaspaceEncode(data.vocab, data.merges, text, data.ids.unk, data.replaces)
    : bpeEncode(data.vocab, data.merges, text);
}

function childNodes(node: unknown): unknown[] {
  if (!node || typeof node !== "object") return [];
  const o = node as Record<string, unknown>;
  const out: unknown[] = [];
  for (const k of ["normalizers", "pre_tokenizers", "decoders"]) {
    const v = o[k];
    if (Array.isArray(v)) out.push(...v);
  }
  return out;
}

/** True when a normalizer/pre-tokenizer/decoder node (or nested Sequence member) has a type. */
function hasNodeType(node: unknown, want: string): boolean {
  if (!node || typeof node !== "object") return false;
  if ((node as Record<string, unknown>)["type"] === want) return true;
  return childNodes(node).some((c) => hasNodeType(c, want));
}

/** Collect normalizer Replace rules ({pattern: {String}, content}) in order. */
function collectReplaces(node: unknown, out: Array<[string, string]>): void {
  if (!node || typeof node !== "object") return;
  const o = node as Record<string, unknown>;
  if (o["type"] === "Replace") {
    const pat = o["pattern"] as Record<string, unknown> | undefined;
    const from = pat?.["String"];
    const to = o["content"];
    if (typeof from === "string" && typeof to === "string") out.push([from, to]);
  }
  for (const c of childNodes(node)) collectReplaces(c, out);
}

/** Parse an HF tokenizer.json ({model vocab/merges, normalizer, pre_tokenizer, added_tokens}). */
export function parseTokenizerJson(raw: unknown): TokenizerData | null {
  try {
    const r = raw as {
      model?: { vocab?: Record<string, number>; merges?: Array<string | [string, string]> };
      normalizer?: unknown;
      pre_tokenizer?: unknown;
      added_tokens?: Array<{ id?: number; content?: string }>;
    };
    const vocabObj = r?.model?.vocab;
    if (!vocabObj || typeof vocabObj !== "object") return null;
    const vocab = new Map(Object.entries(vocabObj));
    const merges = new Map<string, number>();
    for (const [i, m] of (r.model?.merges ?? []).entries()) {
      const pair = typeof m === "string" ? m.split(" ") : m;
      if (pair.length >= 2) merges.set(pair[0] + " " + pair[1], i);
    }
    const added = new Map<string, number>();
    for (const t of r.added_tokens ?? []) {
      if (typeof t?.content === "string" && typeof t?.id === "number") added.set(t.content, t.id);
    }
    const pick = (aliases: readonly string[], fb: number): { id: number; token: string } => {
      for (const a of aliases) {
        const v = added.get(a) ?? vocab.get(a);
        if (v !== undefined) return { id: v, token: a };
      }
      return { id: fb, token: aliases[0] };
    };
    const cls = pick(SPECIAL_ALIASES.cls, CHECKPOINT_IDS.cls);
    const sep = pick(SPECIAL_ALIASES.sep, CHECKPOINT_IDS.sep);
    const mask = pick(SPECIAL_ALIASES.mask, CHECKPOINT_IDS.mask);
    const pad = pick(SPECIAL_ALIASES.pad, CHECKPOINT_IDS.pad);
    const unk = pick(SPECIAL_ALIASES.unk, CHECKPOINT_IDS.unk);
    const kind: PreTokenizerKind = hasNodeType(r?.pre_tokenizer, "Metaspace") ? "metaspace" : "bytelevel";
    const replaces: Array<[string, string]> = [];
    collectReplaces(r?.normalizer, replaces);
    if (kind === "metaspace" && replaces.length === 0) replaces.push([" ", METASPACE_REPLACEMENT]);
    return {
      vocab, merges,
      ids: { cls: cls.id, sep: sep.id, mask: mask.id, pad: pad.id, unk: unk.id },
      kind,
      maskToken: mask.token,
      replaces,
    };
  } catch {
    return null;
  }
}

/** Load an HF tokenizer.json from a local path (node) or URL into vocab/merges/ids. */
export async function loadTokenizerJson(pathOrUrl: string): Promise<TokenizerData | null> {
  let raw: unknown;
  if (/^https?:\/\//.test(pathOrUrl)) {
    const res = await fetch(pathOrUrl);
    if (!res.ok) return null;
    raw = JSON.parse(new TextDecoder().decode(await res.arrayBuffer()));
  } else {
    const fs: typeof import("node:fs/promises") = await import("node:fs/promises");
    raw = JSON.parse(await fs.readFile(pathOrUrl, "utf8"));
  }
  return parseTokenizerJson(raw);
}
