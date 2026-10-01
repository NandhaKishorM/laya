import {
  TEMP_MAX,
  TEMP_MIN,
  buildQuestionPrefix,
  clampTemperature,
  collateItems,
  confidenceFromProbs,
  answerConfidence,
  checkMinConfidence,
  flagLowConfidence,
  renderCriterion,
  renderOptions,
  sequenceWithState,
  serializeState,
  softmax,
  tempBucket,
} from "./common.js";
import type { SequenceStats } from "./common.js";
import type { Batch, SessionProvider } from "./providers.js";
import { encodeWithData, parseTokenizerJson, type TokenizerLike } from "./tokenizer.js";
import { decide, type DecideOptions, type DecisionResult } from "./structured.js";
import {
  HookRegistry,
  PredictContext,
  aggregateUsage,
  composeHooks,
  defaultsAlreadyRan,
  dispatchAsync,
  normaliseHooks,
  type HookArg,
  type PredictHook,
} from "./hooks.js";

export const QTYPES: Record<string, number> = { choice: 0, score: 1, noul: 2 };

export interface QuestionDef {
  type: string;
  instructions?: unknown;
  criteria?: unknown;
  [k: string]: unknown;
}

export interface ActionInfo {
  act_probability: number;
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
  low_confidence?: boolean;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
  low_confidence?: boolean;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
  low_confidence?: boolean;
}

export type SystemAnswer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface SystemUsage {
  input_tokens: number;
  output_tokens: number;
  /** State-fit reporting (issue #174, mirrors Python #181). Present on inference results;
   * absent on hook-supplied (skipped) results and empty-question early returns. */
  state_tokens?: number;
  /** Worst case across the questions, which share one state but not one head budget. */
  state_tokens_dropped?: number;
  truncated?: boolean;
  truncated_questions?: string[];
}

export interface SystemOneResult {
  model: string;
  answers: Record<string, SystemAnswer>;
  usage: SystemUsage;
}

export interface AgentCfg {
  max_len?: number;
  head_max_len?: number;
  temperature?: unknown;
  temperature_by_options?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface AgentOptions {
  provider: SessionProvider;
  tok?: TokenizerLike;
  cfg?: AgentCfg;
  /** Commit SHA the artifacts were loaded from (pinned/requested or `x-repo-commit`); null for local dirs. */
  revision?: string | null;
  max_len?: number;
  head_max_len?: number;
  temperature?: unknown;
  temperature_by_options?: Record<string, unknown>;
  /**
   * Per-language temperature overrides, keyed by language code; keys are normalised to
   * their base subtag (`de-AT` -> `de`), matching Python `Agent(lang_temperatures=...)`.
   * Each entry may carry a `temperature` list of 3 floats (default: the base raw
   * temperature) and/or a `temperature_by_options` map (default: none). A matching
   * override replaces the scale wholesale — see the note at the decode site.
   */
  lang_temperatures?: Record<
    string,
    { temperature?: unknown; temperature_by_options?: Record<string, unknown> } | null
  >;
  hooks?: HookArg;
  onPredictStart?: PredictHook;
  onPredictEnd?: PredictHook;
  hooksRaise?: boolean;
}

/** Per-call options shared by Agent.systemOne/predict and Router.predict. */
export interface PredictOptions {
  /**
   * Language of the request (e.g. "de"); when the Agent has a matching
   * `lang_temperatures` override it selects that language's temperature, exactly like
   * Python `system_one(..., lang=...)`. Routing alone never sets this.
   */
  lang?: string | null;
  hooks?: HookArg;
  onPredictStart?: PredictHook;
  onPredictEnd?: PredictHook;
  hooksRaise?: boolean;
  /** Per-call token-budget overrides; null = agent config. A start hook may also set them. */
  maxLen?: number | null;
  headMaxLen?: number | null;
  /** predictBatch: cap on states per shared forward pass; null sends them all in one pass. */
  batchSize?: number | null;
  /** Minimum confidence threshold in [0.0, 1.0]. Low confidence answers get `low_confidence: true`. */
  minConfidence?: number | null;
  /** Python parity alias for minConfidence. */
  min_confidence?: number | null;
}

function qidStr(qid: string): string {
  return JSON.stringify(qid);
}

export function checkQuestion(qid: string, qdef: unknown): void {
  if (typeof qid !== "string" || !qid.trim()) {
    throw new Error(`question id must be a non-empty string, got ${qidStr(qid)}`);
  }
  if (typeof qdef !== "object" || qdef === null || Array.isArray(qdef)) {
    const got = Array.isArray(qdef) ? "list" : qdef === null ? "NoneType" : typeof qdef;
    throw new Error(`question ${qidStr(qid)}: definition must be a dict, got ${got}`);
  }
  const q = qdef as Record<string, unknown>;
  const t = q["type"];
  if (t !== "choice" && t !== "score" && t !== "noul") {
    throw new Error(
      `question ${qidStr(qid)}: unknown type ${JSON.stringify(t)}; use one of ${JSON.stringify(Object.keys(QTYPES).sort())}`,
    );
  }
  if (!("instructions" in q)) {
    throw new Error(`question ${qidStr(qid)}: no 'instructions'; add the text the model should answer`);
  }
  // `instructions` is the text the model is asked, so a null or empty one is not a weak question
  // but a question with nothing to answer. Only the key's presence was checked, and `toInternal`
  // then serialised whatever it found, so a null reached the model as the literal prompt "null",
  // an empty dict as "{}", and a blank string stayed blank -- the caller's error answered
  // silently. Python's `Agent._check_question` rejects all of these; these are the same three
  // rejections in the same order.
  const ins = q["instructions"];
  if (ins === null || ins === undefined) {
    throw new Error(
      `question ${qidStr(qid)}: 'instructions' must not be None; add the text the model should answer`,
    );
  }
  if (typeof ins === "string" && !ins.trim()) {
    throw new Error(
      `question ${qidStr(qid)}: 'instructions' must not be empty; add the text the model should answer`,
    );
  }
  // An object here is a list or a dict, and both are empty exactly when they own no keys.
  if (typeof ins === "object" && Object.keys(ins as object).length === 0) {
    throw new Error(
      `question ${qidStr(qid)}: 'instructions' must not be empty; add the text the model should answer`,
    );
  }
  // Python allows str, dict, list, int and float. A bool is an int there, so it passes as well;
  // a function or a symbol is nothing either side can turn into a question.
  if (!["string", "object", "number", "bigint", "boolean"].includes(typeof ins)) {
    throw new Error(
      `question ${qidStr(qid)}: 'instructions' must be a string, dict, or list, got ${typeof ins}`,
    );
  }
  const crit = q["criteria"];
  if (t === "choice") {
    if (typeof crit !== "object" || crit === null) {
      throw new Error(
        `question ${qidStr(qid)}: a choice question takes 'criteria' as a dict of label -> description, or a list of labels`,
      );
    }
    if (Object.keys(crit as object).length === 0) {
      throw new Error(`question ${qidStr(qid)}: a choice question needs at least one criterion`);
    }
    if (Array.isArray(crit)) {
      // An indexed loop, not forEach: a hole in a sparse list is an undefined label too.
      for (let i = 0; i < crit.length; i++) {
        const label: unknown = crit[i];
        if (label === null || label === undefined) {
          // A null label becomes the answer key "null", which a client cannot tell apart from
          // the string "null", and `criteria[choice]` never finds it. Python rejects it (#508).
          throw new Error(
            `question ${qidStr(qid)}: choice label ${i} is null; a label is rendered as option text and used ` +
              `as the answer key, so it must be a string, number or bool`,
          );
        }
        if (typeof label !== "object") continue;
        throw new Error(
          `question ${qidStr(qid)}: choice label ${i} is a ${Array.isArray(label) ? "list" : "dict"}; a label is ` +
            `rendered as option text and used as the answer key, so it must be a scalar (a string, number or ` +
            `bool), got ${JSON.stringify(label)}`,
        );
      }
      // toInternal turns the list into an object, so each label's String() is its answer key;
      // two labels on one key would silently drop an option. Python rejects them (#496).
      const seen = new Map<string, number>();
      crit.forEach((label: unknown, i) => {
        const key = String(label);
        const first = seen.get(key);
        if (first === undefined) {
          seen.set(key, i);
          return;
        }
        throw new Error(
          `question ${qidStr(qid)}: choice label ${i} (${JSON.stringify(label)}) repeats label ${first}; the ` +
            `labels are the answer keys, so every option needs its own (1 and "1" are one key)`,
        );
      });
    }
  } else if (t === "score") {
    if (!Array.isArray(crit)) {
      throw new Error(
        `question ${qidStr(qid)}: a score question takes 'criteria' as a list of level descriptions, index 0 first`,
      );
    }
    if (crit.length === 0) {
      throw new Error(`question ${qidStr(qid)}: a score question needs at least one level`);
    }
    const nullAt = crit.findIndex((c) => c === null || c === undefined);
    if (nullAt >= 0) {
      throw new Error(
        `question ${qidStr(qid)}: score level ${nullAt} is null; give every level a description, index 0 first`,
      );
    }
  } else if (crit !== undefined && crit !== null && (typeof crit !== "object" || Array.isArray(crit))) {
    throw new Error(
      `question ${qidStr(qid)}: a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it`,
    );
  } else if (crit && typeof crit === "object" && !Array.isArray(crit)) {
    const invalid = Object.keys(crit as Record<string, unknown>).filter((key) => key.toLowerCase() !== "true" && key.toLowerCase() !== "false");
    if (invalid.length > 0) {
      throw new Error(
        `question ${qidStr(qid)}: noul criteria may contain only 'true' and 'false'; got ${JSON.stringify(invalid)}`,
      );
    }
  }
  if ("labels" in q && t !== "noul") {
    throw new Error(`question ${qidStr(qid)}: 'labels' is only supported for noul questions`);
  }
  if (t === "noul" && q["labels"] !== null && q["labels"] !== undefined) {
    const labels = q["labels"];
    if (typeof labels !== "object" || labels === null || Array.isArray(labels)) {
      throw new Error(`question ${qidStr(qid)}: noul labels must be an object with 'false' and 'true'`);
    }
    const entries = Object.entries(labels as Record<string, unknown>);
    const keys = entries.map(([key]) => key).sort();
    if (keys.length !== 2 || keys[0] !== "false" || keys[1] !== "true" ||
        entries.some(([, value]) => typeof value !== "string" || value.trim() === "")) {
      throw new Error(`question ${qidStr(qid)}: noul labels must map exactly 'false' and 'true' to distinct non-empty strings`);
    }
    const falseLabel = String((labels as Record<string, unknown>)["false"]).trim();
    const trueLabel = String((labels as Record<string, unknown>)["true"]).trim();
    if (falseLabel === trueLabel) {
      throw new Error(`question ${qidStr(qid)}: noul labels must be distinct`);
    }
  }
}

function isPlainDict(o: unknown): o is Record<string, unknown> {
  if (typeof o !== "object" || o === null || Array.isArray(o)) return false;
  const proto = Object.getPrototypeOf(o);
  return proto === null || proto === Object.prototype;
}

/** Whether a `temperature_by_options` value means "no override" in Python.
 *
 * `cfg.get("temperature_by_options") or {}` turns every Python-falsy value into `{}`, and the
 * two languages disagree about which values those are: an empty array is truthy in JavaScript,
 * and NaN is falsy here while truthy in Python. So the set is named rather than inferred from
 * JavaScript truthiness. Python's own falsy values are `[]`, `{}`, `0`, `0.0`, `False`, `None`
 * and `""`; `{}` needs no case here because a dict is accepted either way.
 */
function isEmptyOverride(v: unknown): boolean {
  if (v === null || v === undefined || v === false || v === "") return true;
  if (v === 0) return true; // `===`, not Object.is: that distinguishes -0 from 0
  return Array.isArray(v) && v.length === 0; // [] -- truthy in JavaScript
}

export function toInternal(qdef: QuestionDef): { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } } {
  const t = qdef["type"] as "choice" | "score" | "noul";
  let crit: unknown = qdef["criteria"];
  if (t === "choice" && Array.isArray(crit)) {
    crit = Object.fromEntries(crit.map((c: unknown) => [c as string, null]));
  } else if (t === "noul" && crit !== null && crit !== undefined && typeof crit === "object" && !Array.isArray(crit)) {
    crit = Object.fromEntries(Object.entries(crit as Record<string, unknown>).map(([k, v]) => [String(k).toLowerCase(), v]));
  }
  let ins: unknown = qdef["instructions"];
  if (typeof ins !== "string") ins = serializeState(ins);
  const out: { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } } = {
    t, ins: ins as string, crit,
  };
  if (t === "noul" && qdef["labels"] && typeof qdef["labels"] === "object" && !Array.isArray(qdef["labels"])) {
    const labels = qdef["labels"] as Record<string, unknown>;
    out.labels = { false: String(labels["false"]).trim(), true: String(labels["true"]).trim() };
  }
  return out;
}

export function defaultTokenizer(): TokenizerLike {
  return {
    clsId: 101,
    sepId: 102,
    maskId: 103,
    padId: 0,
    maskToken: "[MASK]",
    encode(text: string): number[] {
      return text
        .split(/\s+/)
        .filter(Boolean)
        .map((w, i) => 1000 + ((w.length * 31 + i * 7) % 20000));
    },
  };
}

function tokenizerFromHF(tokenizerJson: unknown): TokenizerLike | null {
  const data = parseTokenizerJson(tokenizerJson);
  if (!data) return null;
  return {
    clsId: data.ids.cls,
    sepId: data.ids.sep,
    maskId: data.ids.mask,
    padId: data.ids.pad,
    maskToken: data.maskToken,
    encode: (text: string) => encodeWithData(data, text),
  };
}

const r4 = (v: number): number => Math.round(v * 1e4) / 1e4;

export class Agent extends HookRegistry {
  hooksRaise: boolean;
  cfg: AgentCfg;
  provider: SessionProvider;
  revision: string | null;
  tok: TokenizerLike;
  maxLen: number;
  headMaxLen: number;
  temperatureRaw: unknown;
  temperatureByOptionsRaw: Record<string, unknown>;
  temperature: number[];
  temperatureByOptions: Record<string, number>;
  langTemperatures: Record<
    string,
    { temperature: number[]; temperatureByOptions: Record<string, number> }
  >;

  constructor(opts: AgentOptions) {
    super();
    if (!opts || !opts.provider) throw new Error("Agent needs a provider");
    this.provider = opts.provider;
    this.revision = opts.revision ?? null;
    // Hooks are opt-in; an unset hook list is a no-op. See hooks.ts.
    this.hooks = normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    this.hooksRaise = opts.hooksRaise ?? true;
    const cfg = { ...(opts.cfg ?? {}) } as AgentCfg;
    if (opts.max_len !== undefined) cfg.max_len = opts.max_len;
    if (opts.head_max_len !== undefined) cfg.head_max_len = opts.head_max_len;
    if (opts.temperature !== undefined) cfg.temperature = opts.temperature;
    if (opts.temperature_by_options !== undefined) cfg.temperature_by_options = opts.temperature_by_options;
    this.cfg = cfg;
    this.maxLen = Number(cfg.max_len ?? 512);
    this.headMaxLen = Number(cfg.head_max_len ?? 192);
    this.tok = opts.tok ?? defaultTokenizer();
    const raw = (cfg.temperature ?? [1.0, 1.0, 1.0]) as unknown;
    this.temperatureRaw = raw;
    // The decode indexes this by question type, so refuse any other shape here, as Python's
    // Agent.__init__ does (#502): a short list silently fell back to 1.0 per missing type.
    if (!Array.isArray(raw) || raw.length !== 3) {
      throw new Error(`Incompatible model: temperature must be a list of 3 floats, got ${JSON.stringify(raw)}`);
    }
    const rawList: unknown[] = raw;
    this.temperature = rawList.map((t) => clampTemperature(t));
    this.temperatureByOptionsRaw = (cfg.temperature_by_options ?? {}) as Record<string, unknown>;
    this.temperatureByOptions = Object.fromEntries(
      Object.entries(this.temperatureByOptionsRaw).map(([k, v]) => [k, clampTemperature(v)]),
    );
    const entries: Array<[string, unknown, number]> = [
      ...Object.entries(this.temperatureByOptionsRaw).map(
        ([k, v]) => [k, v, this.temperatureByOptions[k]] as [string, unknown, number],
      ),
      ...[0, 1, 2].map(
        (i) => [`temperature[${i}]`, rawList[i], this.temperature[i]] as [string, unknown, number],
      ),
    ];
    const rejected: string[] = [];
    for (const [name, rawV, applied] of entries) {
      if (Number(rawV) === applied) continue;
      rejected.push(`${name}=${JSON.stringify(rawV) ?? String(rawV)} -> ${applied}`);
    }
    if (rejected.length > 0) {
      console.warn(
        `laya: this checkpoint ships invalid temperatures or values outside [${TEMP_MIN}, ${TEMP_MAX}]; ` +
          `using ${rejected.join(", ")}. Treat confidence from the affected entries as uncalibrated.`,
      );
    }
    // Mirrors Agent.__init__ (agent.py): keys normalise to the base subtag, an omitted
    // temperature defaults to the base raw temperature, and every value is clamped.
    this.langTemperatures = {};
    for (const [l, lc] of Object.entries(opts.lang_temperatures ?? {})) {
      const normL = l.split("-")[0].toLowerCase();
      const tRaw = lc?.temperature ?? rawList;
      if (!Array.isArray(tRaw) || tRaw.length !== 3) {
        throw new Error(
          `Language override ${JSON.stringify(l)} temperature must be a list of 3 floats`,
        );
      }
      const tboRaw: unknown = lc?.temperature_by_options;
      // Python: `cfg.get("temperature_by_options") or {}`, then `isinstance(..., dict)`. So a
      // Python-falsy value is an empty override, a dict is kept, and anything else is rejected
      // naming the language. isPlainDict is the module's own mapping test, so a Map, Set or Date
      // is refused instead of being read as {} by Object.entries -- a configured override that
      // silently does nothing is the same failure as no override at all.
      if (!isEmptyOverride(tboRaw) && !isPlainDict(tboRaw)) {
        throw new Error(
          `Language override ${JSON.stringify(l)} temperature_by_options must be a mapping of ` +
            `bucket -> float, got ${Array.isArray(tboRaw) ? "list" : typeof tboRaw}`,
        );
      }
      const tboDict: Record<string, unknown> = isPlainDict(tboRaw) ? tboRaw : {};
      this.langTemperatures[normL] = {
        temperature: [0, 1, 2].map((i) => clampTemperature(tRaw[i])),
        temperatureByOptions: Object.fromEntries(
          Object.entries(tboDict).map(([k, v]) => [k, clampTemperature(v)]),
        ),
      };
    }
  }

  /**
   * Evaluate typed questions across one state in a single forward pass.
   *
   * `hooks` / `onPredictStart` / `onPredictEnd` observe or shape the prediction, appended
   * after any hooks installed on the Agent; a start hook may rewrite the state/questions or
   * call `ctx.skip(...)` to short-circuit inference, an end hook may rewrite the results.
   * See hooks.ts. `hooksRaise` overrides the Agent's setting for this call.
   */
  async systemOne(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: PredictOptions = {},
  ): Promise<SystemOneResult> {
    return (await this._predictHooked([state], questions, opts))[0];
  }

  private async _predictHooked(
    states: unknown[],
    questions: Record<string, QuestionDef>,
    opts: PredictOptions,
  ): Promise<SystemOneResult[]> {
    const mcOpt = opts.minConfidence ?? opts.min_confidence;
    const mc = mcOpt !== undefined && mcOpt !== null ? checkMinConfidence(mcOpt) : null;
    const active = defaultsAlreadyRan(opts)
      ? [...this.hooks, ...normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd)]
      : composeHooks(this.hooks, opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    const raiseErrors = opts.hooksRaise ?? this.hooksRaise;
    const ctx = new PredictContext({
      states,
      questions: questions as Record<string, unknown>,
      agent: this,
      maxLen: opts.maxLen ?? null,
      headMaxLen: opts.headMaxLen ?? null,
    });
    try {
      await dispatchAsync(active, "onPredictStart", ctx, { raiseErrors });
      if (ctx.results === null) {
        if (!Array.isArray(ctx.states)) {
          throw new TypeError(
            "predictBatch expects an array of states; pass a single state to predict()/systemOne().",
          );
        }
        if (!isPlainDict(ctx.questions)) {
          const typeName =
            ctx.questions === null
              ? "NoneType"
              : Array.isArray(ctx.questions)
              ? "list"
              : typeof ctx.questions === "object"
              ? ((ctx.questions as object).constructor?.name ?? "object")
              : typeof ctx.questions;
          throw new TypeError(
            `questions must be a dict of question id -> definition, got ${typeName}`,
          );
        }
        for (const st of ctx.states) {
          if (st === null || st === undefined) {
            throw new TypeError("state must not be None; pass a string, dict, or list");
          }
        }
        const out = await this._systemOneMany(ctx.states, ctx.questions as Record<string, QuestionDef>, {
          maxLen: ctx.maxLen,
          headMaxLen: ctx.headMaxLen,
          batchSize: opts.batchSize ?? null,
          lang: opts.lang ?? null,
        });
        ctx.results = out as unknown as Record<string, unknown>[];
        ctx.model ??= out[0]?.model ?? null;
      }
    } catch (err) {
      ctx.error = err;
      try {
        await dispatchAsync(active, "onError", ctx, { raiseErrors });
      } catch {
        // A failing onError hook must not hide the failure that triggered it.
      }
      throw err;
    } finally {
      ctx.markElapsed();
      if (ctx.results !== null) {
        ctx.usage = aggregateUsage(ctx.results);
        if (mc !== null) {
          flagLowConfidence(ctx.results as unknown as Record<string, unknown>[], mc);
        }
      }
      try {
        await dispatchAsync(active, "onPredictEnd", ctx, { raiseErrors });
      } catch (hookErr) {
        // End hooks run on the failure path too; do not let one mask the real error.
        if (ctx.error === null) throw hookErr;
      }
    }
    return ctx.results as unknown as SystemOneResult[];
  }

  /** Encode one state into one sequence row per question; the state text is tokenized once. */
  private _encodeItems(
    state: unknown,
    ids: string[],
    internals: { t: "choice" | "score" | "noul"; ins: string; crit: unknown }[],
    maxLen: number,
    headMaxLen: number,
  ): { items: { ids: number[]; markers: number[]; qtype: number }[]; nTokens: number; stats: SequenceStats[] } {
    // The state text is shared by every question and is usually the longest text in the
    // sequence — encode it once and compose the per-question prefixes onto it.
    const stAll = this.tok.encode(serializeState(state).split(this.tok.maskToken).join(" "));
    const items: { ids: number[]; markers: number[]; qtype: number }[] = [];
    const stats: SequenceStats[] = [];
    for (let qi = 0; qi < ids.length; qi++) {
      const qid = ids[qi];
      const q = internals[qi];
      const prefix = buildQuestionPrefix(this.tok, q, maxLen, headMaxLen);
      const seq = sequenceWithState(
        prefix,
        stAll,
        this.tok.sepId,
        maxLen,
        Array.isArray(state),
      );
      if (seq.markers.length !== renderOptions(q).length) {
        throw new Error(`question ${qidStr(qid)} options exceed head_max_len=${headMaxLen}`);
      }
      items.push({ ids: seq.ids, markers: seq.markers, qtype: QTYPES[q.t] });
      stats.push(seq.stats);
    }
    return { items, nTokens: items.reduce((a, it) => a + it.ids.length, 0), stats };
  }

  /**
   * Evaluate one shared question schema over many states. Each chunk of up to `batchSize`
   * states is collated into a single encoder+head forward pass; results align with `states`
   * by index. (Python `Agent.predict_batch` parity; its `sort_by_length` is not ported.)
   */
  private async _systemOneMany(
    states: unknown[],
    questions: Record<string, QuestionDef>,
    opts: {
      maxLen?: number | null;
      headMaxLen?: number | null;
      batchSize?: number | null;
      lang?: string | null;
    } = {},
  ): Promise<SystemOneResult[]> {
    if (states.length === 0) return [];
    const ids = Object.keys(questions ?? {});
    if (ids.length === 0) {
      return states.map(() => ({
        model: "laya-rl-agent",
        answers: {},
        usage: { input_tokens: 0, output_tokens: 0 },
      }));
    }
    const maxLen = opts.maxLen ?? this.maxLen;
    const headMaxLen = opts.headMaxLen ?? this.headMaxLen;
    const internals = ids.map((qid) => {
      checkQuestion(qid, questions[qid]);
      return toInternal(questions[qid]);
    });
    const chunkSize =
      opts.batchSize && opts.batchSize > 0 ? Math.floor(opts.batchSize) : states.length;
    const out: SystemOneResult[] = [];
    for (let start = 0; start < states.length; start += chunkSize) {
      const chunk = states.slice(start, start + chunkSize);
      const built = chunk.map((st) => this._encodeItems(st, ids, internals, maxLen, headMaxLen));
      const collated = collateItems(built.map((b) => b.items), this.tok.padId);
      if (!collated) throw new Error("no items to collate");
      const batch: Batch = collated;
      const { lastHidden } = await this.provider.runEncoder(batch);
      const { logits, act } = await this.provider.runHead(lastHidden, batch);
      const rows = chunk.length * ids.length;
      if (!Array.isArray(logits) || !Array.isArray(act) || logits.length < rows || act.length < rows) {
        throw new Error("model provider returned fewer output rows than input items");
      }
      for (let s = 0; s < chunk.length; s++) {
        for (let qi = 0; qi < ids.length; qi++) {
          const r = s * ids.length + qi;
          const k = built[s].items[qi].markers.length;
          const logitRow = logits[r] as number[];
          const actRow = act[r] as number[];
          if (
            !Array.isArray(logitRow) || logitRow.length < k ||
            !Array.isArray(actRow) || actRow.length === 0 ||
            logitRow.some((v) => !Number.isFinite(v)) || actRow.some((v) => !Number.isFinite(v))
          ) {
            throw new Error(`model provider returned invalid output for question ${ids[qi]}`);
          }
        }
      }

      let row = 0;
      for (let s = 0; s < chunk.length; s++) {
        const answers: Record<string, SystemAnswer> = {};
        for (let qi = 0; qi < ids.length; qi++) {
          answers[ids[qi]] = this._decodeRow(
            logits[row] as number[],
            (act[row] as number[]) ?? [1, 0],
            internals[qi],
            built[s].items[qi].markers.length,
            opts.lang ?? null,
          );
          row++;
        }
        // How much of the state actually reached the encoder. Truncation is a token budget, so it
        // moves with maxLen, headMaxLen and each question's rendered head — a caller cannot infer
        // it from the length of the state it sent (issue #174, Python #181 parity).
        const st = built[s].stats;
        const dropped = st.reduce((a, x) => Math.max(a, x.state_tokens_dropped), 0);
        out.push({
          model: "laya-rl-agent",
          answers,
          usage: {
            input_tokens: built[s].nTokens,
            output_tokens: 0,
            state_tokens: st[0]?.state_tokens ?? 0,
            state_tokens_dropped: dropped,
            truncated: dropped > 0,
            truncated_questions: ids.filter((_, qi) => st[qi].truncated),
          },
        });
      }
    }
    return out;
  }

  /** Decode one head row into the answer for its question. */
  private _decodeRow(
    logitsRow: number[],
    actRow: number[],
    q: { t: "choice" | "score" | "noul"; ins: string; crit: unknown },
    k: number,
    lang: string | null = null,
  ): SystemAnswer {
    const qt = QTYPES[q.t];
    const bucket = tempBucket(qt, k);
    // Python parity (_decode_answers): a matching lang override replaces the scale
    // wholesale — its own temperature_by_options first, then its 3-slot temperature —
    // so an override without buckets intentionally ignores the base per-bucket entries.
    const langCfg = lang ? this.langTemperatures[lang.split("-")[0].toLowerCase()] : undefined;
    const scale = langCfg
      ? (langCfg.temperatureByOptions[bucket] ?? langCfg.temperature[qt] ?? 1.0)
      : (this.temperatureByOptions[bucket] ?? this.temperature[qt] ?? 1.0);
    const z = logitsRow.slice(0, k).map((v) => v / scale);
    const p = softmax(z);
    const actP = softmax(actRow.slice(0, Math.max(2, actRow.length)));
    const ext = { act_probability: r4(actP[0]) };
    // Same quantity on every question type (max(p)), so callers can gate across types on
    // one number; `confidence` stays as-is for existing callers (entropy for choice/score).
    const ansConf = r4(answerConfidence(p));
    if (q.t === "choice") {
      const keys = Object.keys(q.crit as Record<string, unknown>);
      let best = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
      return {
        type: "choice",
        choice: keys[best],
        probabilities: Object.fromEntries(keys.map((kk, i) => [kk, r4(p[i] ?? 0)])),
        confidence: r4(confidenceFromProbs(p)),
        answer_confidence: ansConf,
        action: ext,
      };
    }
    if (q.t === "score") {
      const exp = p.reduce((a, v, i) => a + i * v, 0);
      return {
        type: "score",
        score: r4(exp),
        // `renderCriterion`, not the raw `c`: a legend maps an index to the TEXT of a level, and
        // the keys are already strings. A numeric scale written directly used to come back with
        // the caller's own JSON types (`{"0": 1}`), so a client that reads a level as a string
        // had to handle a number, a boolean and null as well -- and `{"0": null}` is not even
        // parseable by a Jev client (#302). Same change as the Python `agent.py` and
        // `onnx_agent.py` legends, so the two runtimes return the same value types.
        legend: Object.fromEntries(
          (q.crit as unknown[]).map((c, i) => [String(i), renderCriterion(c)]),
        ),
        probabilities: Object.fromEntries(p.map((v, i) => [String(i), r4(v)])),
        confidence: r4(confidenceFromProbs(p)),
        answer_confidence: ansConf,
        action: ext,
      };
    }
    const pt = p[1] ?? 0;
    return {
      type: "noul",
      noul: r4(pt),
      confidence: r4(Math.max(pt, 1 - pt)),
      // over two options max(p_true, 1 - p_true) is max(p): identical to confidence here
      answer_confidence: ansConf,
      action: ext,
    };
  }

  async predict(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: PredictOptions = {},
  ): Promise<SystemOneResult> {
    return this.systemOne(state, questions, opts);
  }
  /**
   * Evaluate the same questions over many states, packing each chunk of up to
   * `opts.batchSize` states into one shared forward pass (Python `Agent.predict_batch`
   * parity). This is the throughput path: `systemOne`/`predict` run one state per forward
   * pass, which leaves most of the batch dimension idle on GPU. Results align with
   * `states` by index and are identical in shape to `systemOne`'s output.
   *
   * `opts.maxLen` / `opts.headMaxLen` override the agent config for this call; a start
   * hook may also set `ctx.maxLen` / `ctx.headMaxLen` or call `ctx.skip(results)`.
   * Python's `sort_by_length` grouping is not ported.
   */
  async predictBatch(
    states: unknown[],
    questions: Record<string, QuestionDef>,
    opts: PredictOptions = {},
  ): Promise<SystemOneResult[]> {
    return this._predictHooked(states, questions, opts);
  }

  /**
   * Answer `state` against a JSON schema (or explicit `opts.questions`) and return typed
   * values — see `structured.ts`. Pass exactly one of `schema` or `opts.questions`; other
   * options are forwarded to `predict`.
   */
  async decide(
    state: unknown,
    schema: unknown,
    opts: DecideOptions & PredictOptions & { returnDetails: true },
  ): Promise<DecisionResult>;
  async decide(
    state: unknown,
    schema?: unknown,
    opts?: DecideOptions & PredictOptions,
  ): Promise<Record<string, unknown>>;
  async decide(
    state: unknown,
    schema?: unknown,
    opts: DecideOptions & PredictOptions = {},
  ): Promise<Record<string, unknown> | DecisionResult> {
    return decide(this, state, schema, opts);
  }

  static async load(
    modelDirOrRepo: string,
    opts?: {
      device?: string;
      subfolder?: string | null;
      localDir?: string;
      token?: string | null;
      numThreads?: number;
      /** Per-language temperature overrides; see AgentOptions.lang_temperatures. */
      lang_temperatures?: AgentOptions["lang_temperatures"];
      /** Optional commit SHA/branch/tag to fetch; omitted uses the Hub default and existing cache. */
      revision?: string | null;
      /**
       * Opt-in `{artifact name: SHA-256 hexdigest}` check before any artifact is parsed
       * or executed. A missing artifact or digest mismatch throws and loading is refused.
       */
      expectedSha256?: Record<string, string>;
      signal?: AbortSignal | null;
      onProgress?: ((done: number, total: number, file: string) => void) | null;
    },
  ): Promise<Agent> {
    const sub = opts?.subfolder ?? null;
    const isBrowser =
      typeof (globalThis as unknown as { window?: unknown }).window !== "undefined";
    let cfg: AgentCfg = {};
    let tokenizerJson: unknown | null = null;
    let dir = opts?.localDir ?? modelDirOrRepo;
    let revision: string | null = null;
    let provider: SessionProvider;
    if (isBrowser) {
      const { loadWebBundle, createWebProvider } = await import("./providers.js");
      const bundle = await loadWebBundle(modelDirOrRepo, {
        subfolder: sub,
        revision: opts?.revision,
        expectedSha256: opts?.expectedSha256,
        signal: opts?.signal,
        onProgress: opts?.onProgress,
      });
      cfg = bundle.cfg;
      tokenizerJson = bundle.tokenizerJson;
      dir = bundle.dir;
      revision = bundle.revision;
      provider = await createWebProvider(dir, {
        numThreads: opts?.numThreads,
        expectedSha256: opts?.expectedSha256,
      });
    } else {
      const { loadNodeBundle, createNodeProvider } = await import("./providers.js");
      const bundle = await loadNodeBundle(modelDirOrRepo, {
        subfolder: sub,
        localDir: opts?.localDir,
        token: opts?.token,
        revision: opts?.revision,
        expectedSha256: opts?.expectedSha256,
        signal: opts?.signal,
        onProgress: opts?.onProgress,
      });
      cfg = bundle.cfg;
      tokenizerJson = bundle.tokenizerJson;
      dir = bundle.dir;
      revision = bundle.revision;
      provider = await createNodeProvider(dir, {
        device: opts?.device, numThreads: opts?.numThreads, expectedSha256: opts?.expectedSha256,
      });
    }
    if (!tokenizerJson) {
      throw new Error(
        `Incompatible model: tokenizer.json is missing or invalid in ${JSON.stringify(dir)}`,
      );
    }
    let tok: TokenizerLike;
    try {
      const parsed = tokenizerFromHF(tokenizerJson);
      if (!parsed) throw new Error("unsupported tokenizer.json format");
      tok = parsed;
    } catch (error) {
      throw new Error(
        `Incompatible model: tokenizer.json is missing or invalid in ${JSON.stringify(dir)}: ${String(error)}`,
      );
    }
    return new Agent({ provider, tok, cfg, revision, lang_temperatures: opts?.lang_temperatures });
  }
}
