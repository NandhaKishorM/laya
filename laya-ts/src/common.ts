import type { TokenizerLike } from "./tokenizer.js";
export type QType = "choice" | "score" | "noul";
export interface InternalQ { t: QType; ins: string; crit: unknown; labels?: { false: string; true: string } }
/** Python `json.dumps` for a finite number. JavaScript prints the same shortest round-trip
digits, but Python switches to exponent notation below 1e-4 (JavaScript: below 1e-6) and pads the
exponent to two digits (`5e-05`, not `5e-5`). An integer-valued number prints as before, which
matches a Python int below 1e21: JavaScript cannot tell Python's `49.0` from `49`. */
function pyNumber(v: number): string {
  const [mant, exp] = v.toExponential().split("e");
  const e = Number(exp);
  return e >= -4 ? String(v) : `${mant}e-${String(-e).padStart(2, "0")}`;
}
/** Python `json.dumps(v, ensure_ascii=False)` replica: separators (", ", ": "),
unicode raw, unknown types fall back to undefined (caller applies str()). */
function pyJson(v: unknown): string | undefined {
  if (v === null) return "null";
  if (typeof v === "string" || typeof v === "boolean") return JSON.stringify(v);
  if (typeof v === "number") {
    if (Number.isNaN(v)) return "NaN";
    if (v === Infinity) return "Infinity";
    if (v === -Infinity) return "-Infinity";
    return pyNumber(v);
  }
  if (Array.isArray(v)) return `[${v.map((x) => pyJson(x) ?? "null").join(", ")}]`;
  if (typeof v === "object") {
    const proto = Object.getPrototypeOf(v);
    if (proto !== Object.prototype && proto !== null) return undefined;
    const parts: string[] = [];
    for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
      const s = pyJson(x);
      if (s !== undefined) parts.push(`${JSON.stringify(k)}: ${s}`);
    }
    return `{${parts.join(", ")}}`;
  }
  return undefined;
}
export function serializeState(state: unknown): string {
  if (typeof state === "string") return state;
  return pyJson(state) ?? String(state);
}
/** The text of a criterion: the string itself, else the JSON the model was shown.
 *
 * Exported because `agent.ts` needs it for a `score` answer's `legend`, which maps an index to the
 * text of that level. Putting the raw value in made the response's JSON types depend on the
 * caller's input -- a numeric scale came back as `{"0": 1}`, a boolean as `{"0": true}` and a null
 * as `{"0": null}`, which a Jev client refuses to parse (#302).
 */
export function renderCriterion(v: unknown): string {
  return typeof v === "string" ? v : pyJson(v) ?? String(v);
}
export function renderOptions(q: InternalQ): string[] {
  if (q.t === "choice") {
    const crit = q.crit as Record<string, unknown>;
    return Object.entries(crit).map(([k, v]) =>
      v === null || v === undefined || v === "" ? k : `${k}: ${renderCriterion(v)}`);
  }
  if (q.t === "score") {
    return (q.crit as unknown[]).map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  }
  const crit = (q.crit ?? {}) as Record<string, unknown>;
  const labels = q.labels ?? { false: "false", true: "true" };
  const f = crit["false"], t = crit["true"];
  return [
    labels.false + ": " + (f !== null && f !== undefined && f !== "" ? renderCriterion(f) : "no, the statement does not hold"),
    labels.true + ": " + (t !== null && t !== undefined && t !== "" ? renderCriterion(t) : "yes, the statement holds"),
  ];
}
export interface OptionStats {
  /** Options the question defines, not the markers that survived the sequence clamp. */
  total: number;
  /** How many of them still have a token span of their own. */
  distinct: number;
  /** The per-option cap the head budget forced, or null when none was applied. */
  tokens_per_option: number | null;
}
export interface QuestionPrefix {
  /** [CLS] head [SEP] options [SEP] — the state-independent part of the sequence. */
  ids: number[];
  /** Mask-marker positions (absolute; the prefix sits at the start of the final sequence). */
  markers: number[];
  nOptions: number;
  optionStats: OptionStats;
}
/** The question half of `buildSequence`: everything before the state tokens. Hoisted out so
 * callers asking several questions about the same state can encode the state text only once. */
// Per-tokenizer prefix cache (cap 1k, clear on overflow); repeat triage skips re-encode.
const prefixCache = new WeakMap<object, Map<string, QuestionPrefix>>();
export function buildQuestionPrefix(tok: TokenizerLike, q: InternalQ,
    maxLen = 512, headMaxLen = 192, optionOrder?: number[]): QuestionPrefix {
  let per = prefixCache.get(tok as object);
  if (!per) {
    per = new Map();
    prefixCache.set(tok as object, per);
  }
  // Key on what actually builds the prefix: raw criteria lie (JSON.stringify
  // drops undefined values and throws on BigInt), rendered options don't.
  const key = JSON.stringify([q.t, q.ins, renderOptions(q), maxLen, headMaxLen, optionOrder ?? null]);
  const hit = per.get(key);
  if (hit) return hit;
  const built = buildQuestionPrefixUncached(tok, q, maxLen, headMaxLen, optionOrder);
  if (per.size > 1000) per.clear();
  per.set(key, built);
  return built;
}
function buildQuestionPrefixUncached(tok: TokenizerLike, q: InternalQ,
    maxLen = 512, headMaxLen = 192, optionOrder?: number[]): QuestionPrefix {
  const maskTok = tok.maskToken;
  const opts = renderOptions(q);
  const order = optionOrder ?? opts.map((_, i) => i);
  const ins = String(q.ins).split(maskTok).join(" ");
  let headIds = tok.encode(`${q.t} question: ${ins}`);
  let optIds = order.map((i) =>
    [tok.maskId, ...tok.encode(" " + opts[i].split(maskTok).join(" ")).slice(0, 48)]);
  let budget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  let tokensPerOption: number | null = null;
  if (budget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    budget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
    tokensPerOption = per;
  }
  headIds = headIds.slice(0, Math.max(8, budget));
  const ids = [tok.clsId, ...headIds, tok.sepId];
  const markers: number[] = [];
  for (const o of optIds) { markers.push(ids.length); ids.push(...o); }
  ids.push(tok.sepId);
  // Counted on the capped option ids, before assembly, exactly as Python's `build_head` does:
  // re-slicing the finished sequence cannot close the last option's span, so the last option
  // always looks distinguishable however it collided (#538).
  return {
    ids, markers, nOptions: opts.length,
    optionStats: {
      total: opts.length,
      distinct: new Set(optIds.map((o) => o.join(","))).size,
      tokens_per_option: tokensPerOption,
    },
  };
}
export interface SequenceStats {
  /** Encoded length of the full state, before the window clamp. */
  state_tokens: number;
  /** State tokens that actually reached the encoder after the final maxLen clamp. */
  state_tokens_used: number;
  /** State tokens dropped by the clamp: state_tokens - state_tokens_used. */
  state_tokens_dropped: number;
  /** True when the clamp dropped any state token. */
  truncated: boolean;
  /** What the head budget did to this question's options (Python `build_sequence`'s other stats). */
  options: OptionStats;
}

/** Append pre-encoded state tokens to a question prefix. Identical output to building the
 * whole sequence in one pass, but the state only needs encoding once per state, not once
 * per (state, question) pair.
 *
 * The state is clamped to whatever room the head leaves; `stats` reports that clamp so callers
 * never have to guess it from the character length of what they sent (issue #174, Python #181). */
export function sequenceWithState(prefix: QuestionPrefix, stateIds: number[], sepId: number,
    maxLen = 512, truncateLeft = false): { ids: number[]; markers: number[]; stats: SequenceStats } {
  const room = Math.max(0, maxLen - prefix.ids.length - 1);
  // not stateIds.slice(-room): with no room left, slice(-0) is the whole state rather than none of it
  const kept = truncateLeft ? stateIds.slice(Math.max(0, stateIds.length - room)) : stateIds.slice(0, room);
  const ids = [...prefix.ids, ...kept, sepId].slice(0, maxLen);
  // Count against the final clamp rather than `kept`: the clamp is what actually decided
  // which state tokens reached the encoder.
  const used = Math.max(0, Math.min(kept.length, maxLen - prefix.ids.length));
  return {
    ids,
    markers: prefix.markers.filter((m) => m < maxLen),
    stats: {
      state_tokens: stateIds.length,
      state_tokens_used: used,
      state_tokens_dropped: stateIds.length - used,
      truncated: used < stateIds.length,
      options: prefix.optionStats,
    },
  };
}
/** The questions whose options no longer have a token span each, keyed by question id.
 *
 * `total` is what the question defines, not the number of markers that reached the sequence, so a
 * report cannot say "43 of 43" about a question whose 28 missing options never entered the input
 * at all. Mirrors Python `laya.common.collapsed_options`; empty when nothing collapsed, which is
 * the overwhelming majority of requests, so the agents add the key only when it says something. */
export function collapsedOptions(qids: string[],
    stats: (SequenceStats | undefined)[]): Record<string, OptionStats> {
  const out: Record<string, OptionStats> = {};
  qids.forEach((qid, i) => {
    const s = stats[i]?.options;
    if (s && s.distinct < s.total) out[qid] = { total: s.total, distinct: s.distinct,
      tokens_per_option: s.tokens_per_option };
  });
  return out;
}
export function buildSequence(tok: TokenizerLike, state: unknown, q: InternalQ,
    maxLen = 512, headMaxLen = 192, optionOrder?: number[], truncateLeft = false): { ids: number[]; markers: number[]; stats: SequenceStats } {
  const stAll = tok.encode(serializeState(state).split(tok.maskToken).join(" "));
  return sequenceWithState(buildQuestionPrefix(tok, q, maxLen, headMaxLen, optionOrder), stAll, tok.sepId, maxLen, truncateLeft);
}
export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
export function confidenceFromProbs(p: number[]): number {
  const k = p.length;
  if (k < 2) return 1.0;
  const ent = -p.reduce((a, v) => a + v * Math.log(Math.max(v, 1e-12)), 0);
  return Math.min(1, Math.max(0, 1 - ent / Math.log(k)));
}
export function answerConfidence(p: number[]): number {
  // Probability mass on the reported answer: max(p). This is the quantity temperature
  // scaling fits and the one every calibration figure is computed on, so it is stable
  // across option counts and comparable across question types -- unlike
  // confidenceFromProbs, whose entropy scale moves with k.
  if (p.length < 1) return 1.0;
  return Math.min(1, Math.max(0, Math.max(...p)));
}

function pyRepr(v: unknown): string {
  if (typeof v === "boolean") return v ? "True" : "False";
  if (v === null || v === undefined) return "None";
  if (typeof v === "number") {
    if (Number.isNaN(v)) return "nan";
    if (v === Infinity) return "inf";
    if (v === -Infinity) return "-inf";
    return String(v);
  }
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(pyRepr).join(", ")}]`;
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>).map(
      ([k, val]) => `${pyRepr(k)}: ${pyRepr(val)}`
    );
    return `{${entries.join(", ")}}`;
  }
  return String(v);
}

export type MinConfidenceMap = Record<string, number>;
export type MinConfidence = number | MinConfidenceMap;

const BUCKET_KEY = /^(choice|score|noul):(2|3-5|6-10|11\+)$/;

/**
 * Validate a per-bucket abstention-threshold map (#394).
 *
 * Keys are option-count bucket strings like "choice:2", "choice:3-5", "score:6-10", "noul:2",
 * plus an optional "default". Values are numbers in [0.0, 1.0].
 */
export function checkMinConfidenceMap(m: unknown): MinConfidenceMap {
  if (!m || typeof m !== "object" || Array.isArray(m) || Object.keys(m).length === 0) {
    throw new Error(`a min_confidence map must be a non-empty dict of bucket -> float, got ${pyRepr(m)}`);
  }
  const out: MinConfidenceMap = {};
  for (const [key, val] of Object.entries(m as Record<string, unknown>)) {
    if (key !== "default" && !BUCKET_KEY.test(key)) {
      throw new Error(`min_confidence map keys must be strings like 'choice:3-5', got ${pyRepr(key)}`);
    }
    if (typeof val === "boolean" || typeof val !== "number" || !Number.isFinite(val) || val < 0.0 || val > 1.0) {
      throw new Error(`min_confidence must be a float in [0.0, 1.0], got ${pyRepr(val)}`);
    }
    out[key] = val;
  }
  return out;
}

/**
 * Validate opt-in abstention threshold `min_confidence` (#361, #394).
 *
 * Either a real number in [0.0, 1.0] or a per-bucket mapping. Booleans are rejected.
 */
export function checkMinConfidence(v: unknown): MinConfidence {
  if (v && typeof v === "object" && !Array.isArray(v)) {
    return checkMinConfidenceMap(v);
  }
  if (typeof v === "boolean" || typeof v !== "number" || !Number.isFinite(v) || v < 0.0 || v > 1.0) {
    throw new Error(`min_confidence must be a float in [0.0, 1.0], got ${pyRepr(v)}`);
  }
  return v;
}

export function optionBucket(answer: Record<string, unknown>): string | null {
  const qt = answer.type;
  if (qt !== "choice" && qt !== "score" && qt !== "noul") return null;
  const probs = answer.probabilities;
  let k: number;
  if (probs && typeof probs === "object" && !Array.isArray(probs)) {
    k = Object.keys(probs).length;
  } else if (qt === "noul") {
    k = 2;
  } else {
    return null;
  }
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${qt}:${size}`;
}

export function resolveMinConfidence(
  answer: Record<string, unknown>,
  thresholds: MinConfidenceMap,
  defaultVal: number = 0.0,
): number {
  const key = optionBucket(answer);
  if (key !== null && key in thresholds) {
    return thresholds[key];
  }
  return thresholds.default ?? defaultVal;
}

/**
 * Opt-in abstention marker (#361, #394): flag answers whose confidence falls below `min_confidence`.
 *
 * Reads `answer_confidence` (the calibrated max(p) confidence, invariant to label count k),
 * falling back to `confidence` if `answer_confidence` is absent.
 * The raw answer and confidence stay intact; `low_confidence: true` is added.
 *
 * Supports both a scalar number in [0.0, 1.0] and a per-bucket mapping of thresholds.
 */
export function flagLowConfidence(
  results: Array<Record<string, unknown>> | Record<string, unknown>,
  minConfidence: MinConfidence,
): void {
  const isMap = typeof minConfidence === "object" && minConfidence !== null;
  if (!isMap && minConfidence === 0.0) return;
  const list = Array.isArray(results) ? results : [results];
  for (const res of list) {
    const answers = res && typeof res === "object" ? (res as Record<string, unknown>).answers : null;
    if (!answers || typeof answers !== "object") continue;
    for (const a of Object.values(answers as Record<string, unknown>)) {
      if (!a || typeof a !== "object") continue;
      const ansObj = a as Record<string, unknown>;
      let conf = ansObj.answer_confidence;
      if (conf === undefined || conf === null) {
        conf = ansObj.confidence;
      }
      if (typeof conf === "number" && !Number.isNaN(conf)) {
        const thr = isMap ? resolveMinConfidence(ansObj, minConfidence) : minConfidence;
        if (conf < thr) {
          ansObj.low_confidence = true;
        }
      }
    }
  }
}
export const TEMP_MIN = 0.5, TEMP_MAX = 5.0;
export function clampTemperature(t: unknown): number {
  if (t === null || t === undefined || t === "" || typeof t === "boolean") return 1.0;
  const f = typeof t === "number" ? t : Number(t);
  if (!Number.isFinite(f)) return 1.0;
  return Math.min(TEMP_MAX, Math.max(TEMP_MIN, f));
}
export function tempBucket(qtype: number, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${["choice", "score", "noul"][qtype]}:${size}`;
}

export interface BinningEntry {
  bins: number;
  values: number[];
}

export type BinningMap = Record<string, BinningEntry>;

/**
 * Validate a histogram-binning recalibration map.
 *
 * Keys are option-count bucket strings like "choice:2", "choice:3-5", "score:6-10", "noul:2".
 * Values are objects with an integer `bins >= 1` and a `values` array of length `bins`
 * where each number is in [0.0, 1.0].
 */
export function checkBinningMap(m: unknown): BinningMap {
  if (!m || typeof m !== "object" || Array.isArray(m)) {
    throw new Error(`binning_map must be an object of bucket -> {bins, values}, got ${pyRepr(m)}`);
  }
  const out: BinningMap = {};
  for (const [name, entry] of Object.entries(m as Record<string, unknown>)) {
    if (!BUCKET_KEY.test(name)) {
      throw new Error(`binning_map key ${pyRepr(name)} is not a bucket like "choice:2" or "score:3-5"`);
    }
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(`binning_map[${pyRepr(name)}] must be an object with "bins" and "values", got ${pyRepr(entry)}`);
    }
    const rec = entry as Record<string, unknown>;
    const nBins = rec.bins;
    const values = rec.values;
    if (typeof nBins === "boolean" || typeof nBins !== "number" || !Number.isInteger(nBins) || nBins < 1) {
      throw new Error(`binning_map[${pyRepr(name)}] must have an integer "bins" >= 1, got ${pyRepr(nBins)}`);
    }
    if (!Array.isArray(values) || values.length !== nBins) {
      throw new Error(`binning_map[${pyRepr(name)}] must have "values" of length "bins" (${nBins})`);
    }
    const parsedValues: number[] = [];
    for (const v of values) {
      if (typeof v === "boolean" || typeof v !== "number" || !Number.isFinite(v) || v < 0.0 || v > 1.0) {
        throw new Error(`binning_map[${pyRepr(name)}] values must be numbers in [0, 1], got ${pyRepr(v)}`);
      }
      parsedValues.push(v);
    }
    out[name] = { bins: nBins, values: parsedValues };
  }
  return out;
}

/**
 * Recalibrate one `answer_confidence` for its option-count `bucket` (tempBucket).
 *
 * Returns the confidence unchanged when the map has no entry for the bucket, so a bucket the map
 * was not fit for passes through rather than being forced to a wrong value.
 */
export function applyBinningMap(
  confidence: number,
  bucket: string,
  binningMap?: BinningMap | null,
): number {
  if (!Number.isFinite(confidence)) return confidence;
  const entry = binningMap?.[bucket];
  if (!entry) {
    return confidence;
  }
  const bins = entry.bins;
  const b = Math.min(bins - 1, Math.max(0, Math.floor(confidence * bins)));
  return entry.values[b];
}
/** Max of a length list without spread (Math.max(...arr) throws RangeError past ~100k args). */
export function maxOf(values: ArrayLike<number>, fallback = 0): number {
  let m = fallback;
  for (let i = 0; i < values.length; i++) if (values[i] > m) m = values[i];
  return m;
}
export interface CollateItem {
  ids: number[];
  markers: number[];
  qtype: number;
  label?: number;
  target?: number[];
  [k: string]: unknown;
}
export interface CollatedBatch {
  inputIds: number[][];
  attentionMask: number[][];
  markerPos: number[][];
  markerMask: boolean[][];
  qtype: number[];
  label: number[];
  meta: Record<string, unknown>[];
  target?: number[][];
}
/** TS parity of py `collate_items(batch, pad_id)`: batch = list of groups. */
export function collateItems(batch: CollateItem[][], padId: number): CollatedBatch | null {
  const items = (batch ?? []).flat();
  if (items.length === 0) return null;
  let L = 0;
  let K = 0;
  for (const it of items) {
    if (it.ids.length > L) L = it.ids.length;
    if (it.markers.length > K) K = it.markers.length;
  }
  const hasTarget = items.some((it) => "target" in it);
  const inputIds = items.map((it) => [...it.ids, ...Array(L - it.ids.length).fill(padId)]);
  const attentionMask = items.map((it) => [...Array(it.ids.length).fill(1), ...Array(L - it.ids.length).fill(0)]);
  const markerPos = items.map((it) => [...it.markers, ...Array(K - it.markers.length).fill(0)]);
  const markerMask = items.map((it) => [...it.markers.map(() => true), ...Array(K - it.markers.length).fill(false)]);
  const qtype = items.map((it) => it.qtype);
  const label = items.map((it) => (typeof it.label === "number" ? it.label : -1));
  const meta = items.map((it) => {
    const { ids: _ids, markers: _markers, target: _target, ...m } = it;
    return m;
  });
  const out: CollatedBatch = { inputIds, attentionMask, markerPos, markerMask, qtype, label, meta };
  if (hasTarget) {
    out.target = items.map((it, i) => {
      const t = Array.isArray(it.target) ? (it.target as number[]) : [];
      const k = it.markers.length;
      if (t.length > k) {
        // Same guard as py collate_items: the limit is this item's own marker count, not the
        // batch-wide K — a wider sibling row must not legitimise extra entries (#311).
        throw new Error(
          `collateItems: item ${i} has ${t.length} target entries but only ${k} marker positions; ` +
            "a target needs one entry per option",
        );
      }
      return [...t, ...Array(K - t.length).fill(0)];
    });
  }
  return out;
}
