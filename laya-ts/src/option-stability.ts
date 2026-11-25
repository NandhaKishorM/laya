import { serializeState } from "./common.js";
import { BaseHook, type PredictContext } from "./hooks.js";
import { StabilityRandom } from "./_option-stability-random.js";

export type StabilityProbe = "reorder" | "rename";
export type ReliabilityDecision = "ACCEPT" | "VERIFY" | "ESCALATE";

export interface StabilityVariant {
  name: string;
  order: string[];
  relabel: boolean;
}

export interface VariantOptions {
  nShuffles?: number;
  relabel?: boolean;
  include?: readonly string[] | null;
  reorder?: boolean;
}

export interface CalibrationOptions {
  targetAccuracy?: number;
  maxEscalateAccuracy?: number;
}

export interface OptionStabilityOptions {
  /** Domain-calibrated thresholds; pass null for uncalibrated metrics only. */
  accept: number | null;
  escalate: number | null;
  probes?: "auto" | "all" | StabilityProbe | readonly StabilityProbe[] | null;
  nShuffles?: number;
  relabel?: boolean;
  renameMaxOptions?: number;
  variants?: "full" | "fast" | readonly string[] | null;
  /** Use an explicit seed for cross-language parity with structured states. */
  seed?: string | null;
}

export interface OptionStabilityReport {
  probes: StabilityProbe[];
  option_layout: "parallel" | "sequential";
  decision: ReliabilityDecision | null;
  /** The unrounded mean used by decideReliability and calibrateReliability. */
  soft_stability: number | null;
  stability: number | null;
  rename_stability: number | null;
  reorder_stability: number | null;
  n_variants: number;
  distinct_choices: string[];
  variant_choices: Record<string, string | null>;
  variant_support: Record<string, number>;
}

const SEP = "::rel";
const ALL_VARIANTS = ["reversed", "shuffle1", "shuffle2", "shuffle3", "letters", "letters_reversed", "letters_shuffle"];

/** Deterministic option permutations, including the original, with duplicates removed.
 * The CRC32 / Python MT19937 sequence matches laya.hooks.make_variants. Filtering
 * happens after shuffling so a named subset retains the full preset's permutations. */
export function makeVariants(labels: readonly string[], seedKey: string, opts: VariantOptions = {}): StabilityVariant[] {
  const { nShuffles = 3, relabel = true, include = null, reorder = true } = opts;
  checkShuffles(nShuffles);
  const rng = new StabilityRandom(seedKey);
  const candidates: StabilityVariant[] = [{ name: "original", order: [...labels], relabel: false }];
  if (reorder) {
    candidates.push({ name: "reversed", order: [...labels].reverse(), relabel: false });
    for (let i = 0; i < nShuffles; i++) {
      const order = [...labels];
      rng.shuffle(order);
      candidates.push({ name: `shuffle${i + 1}`, order, relabel: false });
    }
  }
  if (relabel && labels.length <= 26) {
    candidates.push({ name: "letters", order: [...labels], relabel: true });
    candidates.push({ name: "letters_reversed", order: [...labels].reverse(), relabel: true });
    const order = [...labels];
    rng.shuffle(order);
    candidates.push({ name: "letters_shuffle", order, relabel: true });
  }
  const wanted = include === null ? null : new Set(include);
  const seen = new Set<string>();
  return candidates.filter((v) => {
    if (wanted && v.name !== "original" && !wanted.has(v.name)) return false;
    const key = JSON.stringify([v.order, v.relabel]);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/** Uncalibrated thresholds fail closed; equality with accept accepts, while equality
 * with escalate verifies. A fitted accept of 1.01 accepts no probability score. */
export function decideReliability(softStability: number, accept: number | null,
    escalate: number | null = null): ReliabilityDecision | null {
  checkProbability(softStability, "softStability");
  checkThresholds(accept, escalate);
  if (accept === null) return null;
  if (softStability >= accept) return "ACCEPT";
  if (escalate !== null && softStability < escalate) return "ESCALATE";
  return "VERIFY";
}

/** Fit [accept, escalate] on labelled examples. Distinct raw scores are candidate
 * cutoffs; equal scores stay together and report JSON can be replayed without rounding.
 * Inputs must be equally sized, non-empty score/label arrays. */
export function calibrateReliability(scores: readonly number[], correct: readonly boolean[],
    opts: CalibrationOptions = {}): [number, number] {
  const { targetAccuracy = 0.90, maxEscalateAccuracy = 1 / 3 } = opts;
  if (scores.length !== correct.length) throw new Error("scores and correct must have the same length");
  if (!scores.length) throw new Error("cannot calibrate reliability on empty scores/correct pairs");
  checkProbability(targetAccuracy, "targetAccuracy");
  checkProbability(maxEscalateAccuracy, "maxEscalateAccuracy");
  const pairs = Array.from(scores, (score, i) => {
    checkProbability(score, `scores[${i}]`);
    if (typeof correct[i] !== "boolean") throw new TypeError(`correct[${i}] must be a bool`);
    return [score, correct[i]] as const;
  });
  const cuts = [...new Set(pairs.map(([score]) => score))].sort((a, b) => a - b);
  let accept = 1.01;
  for (const t of cuts) {
    const kept = pairs.filter(([s]) => s >= t);
    if (kept.length && kept.filter(([, c]) => c).length / kept.length >= targetAccuracy) {
      accept = t;
      break;
    }
  }
  let escalate = 0;
  for (const t of cuts) {
    if (t >= accept) break;
    const below = pairs.filter(([s]) => s < t);
    if (below.length && below.filter(([, c]) => c).length / below.length <= maxEscalateAccuracy) escalate = t;
  }
  return [accept, escalate];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function checkShuffles(value: number): void {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new Error("nShuffles must be a non-negative integer");
  }
}

function checkThreshold(value: number | null, name: string, max: number): void {
  if (value !== null && (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > max)) {
    throw new Error(`${name} must be null or a number in [0, ${max}]`);
  }
}

function checkProbability(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${name} must be a finite number in [0, 1]`);
  }
}

function checkThresholds(accept: number | null, escalate: number | null): void {
  checkThreshold(accept, "accept", 1.01);
  checkThreshold(escalate, "escalate", 1);
  if (accept !== null && escalate !== null && escalate > accept) {
    throw new Error("escalate must not be greater than accept");
  }
}

function resolveProbes(probes: OptionStabilityOptions["probes"]): StabilityProbe[] | null {
  if (probes === undefined || probes === null || probes === "auto") return null;
  if (probes === "all") return ["reorder", "rename"];
  if (probes === "rename" || probes === "reorder") return [probes];
  if (!Array.isArray(probes) || !probes.length || probes.some((p) => p !== "reorder" && p !== "rename")) {
    throw new Error("probes must be 'auto', 'all', 'reorder', 'rename', or a non-empty sequence of probe names");
  }
  return [...new Set(probes)];
}

function resolveVariants(variants: OptionStabilityOptions["variants"]): string[] | null {
  if (variants === undefined || variants === null || variants === "full") return null;
  if (variants === "fast") return ["reversed", "letters_reversed"];
  if (!Array.isArray(variants) || !variants.length || variants.some((v) => !ALL_VARIANTS.includes(v))) {
    throw new Error(`variants must be 'full', 'fast', or a non-empty sequence from ${ALL_VARIANTS.join(", ")}`);
  }
  return [...variants];
}

function detectLayout(ctx: PredictContext): "parallel" | "sequential" {
  const cfg = isRecord(ctx.agent) && isRecord(ctx.agent.cfg) ? ctx.agent.cfg : null;
  const raw = cfg?.option_layout;
  if (typeof raw === "string") {
    const layout = raw.trim().toLowerCase();
    if (layout === "parallel" || layout === "sequential") return layout;
    console.warn(`laya: unknown option_layout ${JSON.stringify(raw)} in agent config; defaulting to 'sequential'`);
  }
  return "sequential";
}

function variantQuestion(q: Record<string, unknown>, order: string[], relabel: boolean):
    { question: Record<string, unknown>; back: Map<string, string> } {
  const shown = relabel ? order.map((_, i) => String.fromCharCode(65 + i)) : [...order];
  const criteria = q.criteria;
  const entries = shown.map((label, i) => {
    const original = order[i];
    let description = Array.isArray(criteria) ? null : (criteria as Record<string, unknown>)[original];
    if (relabel && (description === null || description === undefined ||
        (typeof description === "string" && description.trim() === ""))) description = original;
    return [label, description] as const;
  });
  const nextCriteria = Object.fromEntries(entries);
  const question = { ...q, criteria: Array.isArray(criteria) && !relabel ? [...order] : nextCriteria };
  // Integer-like keys enumerate numerically in JS regardless of insertion order. The
  // encoder already supports a slot permutation, so preserve the requested physical
  // order there, including the caller's permutation of each generated variant.
  const actual = Object.keys(nextCriteria);
  const callerOrder = Array.isArray(q.option_order) ? q.option_order as number[] : order.map((_, i) => i);
  const effectiveOrder = callerOrder.map((i) => actual.indexOf(shown[i]));
  if (Object.hasOwn(q, "option_order") || effectiveOrder.some((i, slot) => i !== slot)) {
    (question as Record<string, unknown>).option_order = effectiveOrder;
  }
  return { question, back: new Map(shown.map((label, i) => [label, order[i]])) };
}

/** Python round(value, 4): round the binary value itself, with ties to even.
 * Multiplying a double by 1e4 first can manufacture a tie, and toFixed ties away. */
function round4(value: number): number {
  if (!Number.isFinite(value) || value === 0) return value;
  const data = new DataView(new ArrayBuffer(8));
  data.setFloat64(0, value);
  const bits = data.getBigUint64(0);
  const exponent = Number((bits >> 52n) & 0x7ffn);
  const mantissa = (bits & ((1n << 52n) - 1n)) | (exponent ? 1n << 52n : 0n);
  const power = exponent ? exponent - 1075 : -1074;
  let numerator = mantissa * 10000n;
  const denominator = power < 0 ? 1n << BigInt(-power) : 1n;
  if (power > 0) numerator <<= BigInt(power);
  let rounded = numerator / denominator;
  const remainder = numerator % denominator;
  if (remainder * 2n > denominator || (remainder * 2n === denominator && (rounded & 1n))) rounded++;
  return (bits >> 63n ? -Number(rounded) : Number(rounded)) / 10000;
}

function compareCodePoints(a: string, b: string): number {
  const x = Array.from(a, (c) => c.codePointAt(0)!);
  const y = Array.from(b, (c) => c.codePointAt(0)!);
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
}

interface PlannedVariant {
  name: string;
  key: string;
  back: Map<string, string>;
  relabel: boolean;
}
interface QuestionPlan {
  layout: "parallel" | "sequential";
  variants: PlannedVariant[];
  probes: StabilityProbe[];
}

/** Opt-in stability probes, batched with the original choice questions. Metrics are
 * signals for calibration, not correctness probabilities. Layout detection selects
 * probes; it does not add parallel-layout support to a provider. */
export class OptionStabilityHook extends BaseHook {
  private readonly accept: number | null;
  private readonly escalate: number | null;
  private readonly probes: StabilityProbe[] | null;
  private readonly nShuffles: number;
  private readonly relabel: boolean;
  private readonly renameMaxOptions: number;
  private readonly include: string[] | null;
  private readonly seed: string | null;
  private readonly plans = new WeakMap<PredictContext, Map<string, QuestionPlan>>();

  constructor(opts: OptionStabilityOptions) {
    super();
    if (!opts || !Object.hasOwn(opts, "accept") || !Object.hasOwn(opts, "escalate")) {
      throw new Error("accept and escalate are required; pass null for uncalibrated metrics");
    }
    checkThresholds(opts.accept, opts.escalate);
    this.accept = opts.accept;
    this.escalate = opts.escalate;
    this.probes = resolveProbes(opts.probes);
    this.nShuffles = opts.nShuffles === undefined ? 3 : opts.nShuffles;
    checkShuffles(this.nShuffles);
    this.relabel = opts.relabel === undefined ? true : opts.relabel;
    if (typeof this.relabel !== "boolean") throw new TypeError("relabel must be a bool");
    this.renameMaxOptions = opts.renameMaxOptions === undefined ? 20 : opts.renameMaxOptions;
    if (!Number.isInteger(this.renameMaxOptions) || this.renameMaxOptions < 2) {
      throw new Error("renameMaxOptions must be an integer >= 2");
    }
    this.include = resolveVariants(opts.variants);
    this.seed = opts.seed === undefined || opts.seed === null ? null : String(opts.seed);
  }

  onPredictStart(ctx: PredictContext): void {
    this.plans.delete(ctx);
    for (const qid of Object.keys(ctx.questions)) {
      if (qid.includes(SEP)) throw new Error(`question id ${JSON.stringify(qid)} must not contain ${JSON.stringify(SEP)}`);
    }
    const layout = detectLayout(ctx);
    const configured = new Set(this.probes ?? (layout === "parallel"
      ? (this.relabel ? ["rename"] : []) : (this.relabel ? ["reorder", "rename"] : ["reorder"])));
    // Python uses repr for structured states; TS keeps its existing serializer.
    // Explicit seed, string states, and multi-state "batch" seeds are shared.
    const baseSeed = this.seed ?? (ctx.states.length === 1 ? serializeState(ctx.states[0]) : "batch");
    const plan = new Map<string, QuestionPlan>();
    const questions = { ...ctx.questions };
    for (const [qid, q] of Object.entries(ctx.questions)) {
      if (!isRecord(q) || q.type !== "choice") continue;
      const criteria = q.criteria;
      if (!Array.isArray(criteria) && !isRecord(criteria)) continue;
      // Match the engine's list-to-dict answer keys, including numeric-like labels.
      const labels = Object.keys(Array.isArray(criteria)
        ? Object.fromEntries(criteria.map((label) => [String(label), null])) : criteria);
      const variants: PlannedVariant[] = [{ name: "original", key: qid,
        back: new Map(labels.map((label) => [label, label])), relabel: false }];
      const probes: StabilityProbe[] = [];
      if (labels.length >= 2) {
        const generated = makeVariants(labels, `${baseSeed}|${qid}`, {
          nShuffles: this.nShuffles,
          relabel: configured.has("rename") && labels.length <= this.renameMaxOptions,
          reorder: configured.has("reorder"), include: this.include,
        }).slice(1);
        if (generated.some((v) => !v.relabel)) probes.push("reorder");
        if (generated.some((v) => v.relabel)) probes.push("rename");
        generated.forEach((variant, i) => {
          const key = `${qid}${SEP}${i + 1}`;
          const { question, back } = variantQuestion(q, variant.order, variant.relabel);
          questions[key] = question;
          variants.push({ name: variant.name, key, back, relabel: variant.relabel });
        });
      }
      plan.set(qid, { layout, variants, probes });
    }
    if (plan.size) {
      ctx.questions = questions;
      this.plans.set(ctx, plan);
    }
  }

  onPredictEnd(ctx: PredictContext): void {
    const plan = this.plans.get(ctx);
    this.plans.delete(ctx);
    if (!plan || !ctx.results) return;
    const internalKeys = new Set([...plan.values()].flatMap((p) => p.variants.slice(1).map((v) => v.key)));
    for (const result of ctx.results) {
      if (!isRecord(result)) continue;
      const answers = result.answers;
      if (isRecord(answers)) {
        for (const [qid, p] of plan) {
          if (!Object.hasOwn(answers, qid)) continue;
          const answer = answers[qid];
          if (!isRecord(answer) || answer.type !== "choice" || typeof answer.choice !== "string") continue;
          const seen: { name: string; choice: string | null; support: number; relabel: boolean }[] = [];
          for (const variant of p.variants) {
            if (!Object.hasOwn(answers, variant.key)) break;
            const value = answers[variant.key];
            if (!isRecord(value) || value.type !== "choice" || typeof value.choice !== "string" ||
                !isRecord(value.probabilities)) break;
            const probabilities = Object.entries(value.probabilities);
            if (probabilities.some(([, v]) => typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1)) break;
            const probs = new Map(probabilities.map(([label, probability]) => [variant.back.get(label) ?? label, probability as number]));
            const choice = variant.back.get(value.choice) ?? value.choice;
            seen.push({ name: variant.name, choice, support: probs.get(answer.choice) ?? 0, relabel: variant.relabel });
          }
          // A partial or malformed answer set cannot establish stability.
          if (seen.length !== p.variants.length) continue;
          const others = seen.slice(1);
          const probed = others.length > 0 && p.probes.length > 0;
          const rate = (items: typeof seen): number | null => items.length
            ? round4(items.filter((v) => v.choice === answer.choice).length / items.length) : null;
          const soft = probed ? seen.reduce((sum, v) => sum + v.support, 0) / seen.length : null;
          const report: OptionStabilityReport = {
            probes: probed ? [...p.probes] : [], option_layout: p.layout,
            decision: soft === null ? null : decideReliability(soft, this.accept, this.escalate),
            soft_stability: soft, stability: probed ? rate(others) : null,
            rename_stability: probed ? rate(others.filter((v) => v.relabel)) : null,
            reorder_stability: probed ? rate(others.filter((v) => !v.relabel)) : null,
            n_variants: seen.length,
            distinct_choices: [...new Set(seen.flatMap((v) => v.choice === null ? [] : [v.choice]))].sort(compareCodePoints),
            variant_choices: Object.fromEntries(seen.map((v) => [v.name, v.choice])),
            variant_support: Object.fromEntries(seen.map((v) => [v.name, round4(v.support)])),
          };
          answer.reliability = report;
        }
        for (const key of internalKeys) delete answers[key];
      }
      const usage = result.usage;
      if (isRecord(usage)) {
        if (Array.isArray(usage.truncated_questions)) {
          usage.truncated_questions = usage.truncated_questions.filter((key) => !internalKeys.has(key));
        }
        if (isRecord(usage.options)) {
          usage.options = Object.fromEntries(Object.entries(usage.options).filter(([key]) => !internalKeys.has(key)));
        }
      }
    }
  }

  onError(ctx: PredictContext): void {
    this.plans.delete(ctx);
  }
}
