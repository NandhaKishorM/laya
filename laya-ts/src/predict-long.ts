import { buildQuestionPrefix, serializeState } from "./common.js";
import type { OptionStats } from "./common.js";
import { checkQuestion, toInternal } from "./agent.js";
import type { Agent, QuestionDef, SystemAnswer, PredictOptions } from "./agent.js";
import { defaultsAlreadyRan, markDefaultsRan, normaliseHooks } from "./hooks.js";
import type { HookArg, PredictContext, PredictHook } from "./hooks.js";

export interface LongWindow {
  index: number;
  token_start: number;
  token_end: number;
  count: number;
}

export type LongAnswer = SystemAnswer & { window?: LongWindow };

/** Long scans sum numeric usage, including boolean truncation flags. */
export interface LongUsage {
  input_tokens: number;
  output_tokens: number;
  windows: number;
  state_tokens?: number;
  state_tokens_dropped?: number;
  truncated?: boolean | number;
  truncated_questions?: string[];
  options?: Record<string, OptionStats>;
  [key: string]: unknown;
}

export interface LongResult {
  model: string;
  answers: Record<string, LongAnswer>;
  usage: LongUsage;
}

// Deliberately independent of PredictOptions: windows are sized from the Agent config,
// and the winning window's confidence is not a document-level confidence threshold.
export interface LongPredictOptions {
  window?: number | null;
  stride?: number | null;
  aggregate?: "auto";
  batchSize?: number | null;
  lang?: string | null;
  hooks?: HookArg;
  onPredictStart?: PredictHook;
  onPredictEnd?: PredictHook;
  hooksRaise?: boolean;
  hooksTimeout?: number | null;
}

type Dict = Record<string, unknown>;
function isDict(value: unknown): value is Dict {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function copyValue<T>(value: T): T {
  if (Array.isArray(value)) return value.map(copyValue) as T;
  if (isDict(value)) return Object.fromEntries(Object.entries(value).map(([key, val]) => [key, copyValue(val)])) as T;
  return value;
}

function sameValue(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((value, i) => sameValue(value, b[i]));
  }
  if (isDict(a) && isDict(b)) {
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((key) => Object.hasOwn(b, key) && sameValue(a[key], b[key]));
  }
  return false;
}

/** Validate values supplied by hooks without widening ordinary prediction types. */
export function validateLongResult(value: unknown, label = "predictLong result", requireWindows = true): asserts value is LongResult {
  if (!isDict(value) || typeof value.model !== "string" || !isDict(value.answers) || !isDict(value.usage)) {
    throw new TypeError(`${label}: expected model, answers and usage objects`);
  }
  const usage = value.usage;
  for (const key of ["input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped"]) {
    if (key !== "input_tokens" && key !== "output_tokens" && usage[key] === undefined) continue;
    if (typeof usage[key] !== "number" || !Number.isFinite(usage[key]) || usage[key] < 0) {
      throw new TypeError(`${label}: usage.${key} must be a finite non-negative number`);
    }
  }
  if ((requireWindows || usage.windows !== undefined) &&
      (!Number.isSafeInteger(usage.windows) || (usage.windows as number) < 0)) {
    throw new TypeError(`${label}: usage.windows must be a non-negative safe integer`);
  }
  if (usage.truncated !== undefined && typeof usage.truncated !== "boolean" &&
      (typeof usage.truncated !== "number" || !Number.isFinite(usage.truncated) || usage.truncated < 0)) {
    throw new TypeError(`${label}: usage.truncated must be a boolean or non-negative number`);
  }
  if (usage.truncated_questions !== undefined && (!Array.isArray(usage.truncated_questions) ||
      usage.truncated_questions.some((qid) => typeof qid !== "string"))) {
    throw new TypeError(`${label}: usage.truncated_questions must be an array of question ids`);
  }
  if (usage.options !== undefined && (!isDict(usage.options) || Object.values(usage.options).some((stats) =>
    !isDict(stats) || ["total", "distinct"].some((key) => typeof stats[key] !== "number" || !Number.isFinite(stats[key]) || stats[key] < 0) ||
    (stats.tokens_per_option !== null && (typeof stats.tokens_per_option !== "number" || !Number.isFinite(stats.tokens_per_option) || stats.tokens_per_option < 0))))) {
    throw new TypeError(`${label}: usage.options must map question ids to option statistics`);
  }
  for (const [qid, answer] of Object.entries(value.answers)) {
    if (!isDict(answer) || !["choice", "score", "noul"].includes(answer.type as string)) {
      throw new TypeError(`${label}: answer ${JSON.stringify(qid)} must have a supported type`);
    }
    for (const key of ["confidence", "answer_confidence", ...(answer.type === "noul" ? ["noul"] : answer.type === "score" ? ["score"] : [])]) {
      if (typeof answer[key] !== "number" || !Number.isFinite(answer[key])) {
        throw new TypeError(`${label}: answer ${JSON.stringify(qid)}.${key} must be a finite number`);
      }
    }
    if (answer.type === "choice" && typeof answer.choice !== "string") {
      throw new TypeError(`${label}: answer ${JSON.stringify(qid)}.choice must be a string`);
    }
    if (!isDict(answer.action) || typeof answer.action.act_probability !== "number" || !Number.isFinite(answer.action.act_probability)) {
      throw new TypeError(`${label}: answer ${JSON.stringify(qid)}.action.act_probability must be a finite number`);
    }
    if (answer.type !== "noul" && (!isDict(answer.probabilities) ||
        Object.values(answer.probabilities).some((probability) => typeof probability !== "number" || !Number.isFinite(probability)))) {
      throw new TypeError(`${label}: answer ${JSON.stringify(qid)}.probabilities must map labels to finite numbers`);
    }
    if (answer.type === "score" && (!isDict(answer.legend) || Object.values(answer.legend).some((level) => typeof level !== "string"))) {
      throw new TypeError(`${label}: answer ${JSON.stringify(qid)}.legend must map levels to strings`);
    }
    if (answer.window !== undefined) {
      const w = answer.window;
      if (!isDict(w) || !["index", "token_start", "token_end", "count"].every((key) => Number.isSafeInteger(w[key]) && (w[key] as number) >= 0) ||
          (w.index as number) >= (w.count as number) || (w.token_start as number) > (w.token_end as number)) {
        throw new TypeError(`${label}: answer ${JSON.stringify(qid)}.window is invalid`);
      }
    }
  }
}

/** Cache results are copied; counting a skipped scan must not mutate the caller's cache. */
export function skippedLongResult(value: unknown): LongResult {
  validateLongResult(value, "predictLong hook result", false);
  return { ...value, usage: { ...value.usage, windows: 0 } };
}

export function validateLongOptions(opts: LongPredictOptions): void {
  for (const key of ["window", "stride", "batchSize"] as const) {
    const value = opts[key];
    if (value !== undefined && value !== null && (!Number.isSafeInteger(value) || value <= 0)) {
      throw new TypeError(`predictLong: ${key} must be a positive safe integer`);
    }
  }
  if (opts.aggregate !== undefined && opts.aggregate !== "auto") {
    throw new Error("predictLong: only aggregate='auto' is supported");
  }
}

interface AnswerRecord { index: number; values: Dict }
interface Evidence {
  answered: boolean;
  states: unknown[] | null;
  questionTypes: Record<string, unknown> | null;
  questions: Dict | null;
  maxLen: number | null;
  headMaxLen: number | null;
  answers: Map<object, Map<string, AnswerRecord | null>> | null;
  resultCount: number | null;
  trackAnswers: boolean;
}

const recorders = new WeakMap<PredictOptions, Evidence>();
export function isLongCall(opts: PredictOptions): boolean { return recorders.has(opts); }

/** The short public systemOne path returns just the first result; retain its final count. */
export function finishLongResults(opts: PredictOptions, ctx: PredictContext): void {
  const evidence = recorders.get(opts);
  if (!evidence) return;
  if (!Array.isArray(ctx.results)) throw new TypeError("predictLong: prediction hooks must return an array of results");
  evidence.resultCount = ctx.results.length;
}
const predictionFields = ["type", "choice", "noul", "score", "legend", "probabilities", "confidence", "answer_confidence"];
function answerValues(answer: Dict): Dict {
  return Object.fromEntries(predictionFields.filter((key) => Object.hasOwn(answer, key)).map((key) => [key, copyValue(answer[key])]));
}

/** Internal dispatcher boundary: record inference results before any end hook can edit them. */
export function recordLongResults(opts: PredictOptions, ctx: PredictContext): void {
  const evidence = recorders.get(opts);
  if (!evidence || !Array.isArray(ctx.results)) return;
  evidence.resultCount = ctx.results.length;
  if (evidence.answered || !evidence.trackAnswers) return;
  const recorded: NonNullable<Evidence["answers"]> = new Map();
  ctx.results.forEach((result, index) => {
    if (!isDict(result?.answers)) return;
    for (const [qid, answer] of Object.entries(result.answers)) {
      if (!isDict(answer)) continue;
      let byQuestion = recorded.get(answer);
      if (!byQuestion) recorded.set(answer, byQuestion = new Map());
      byQuestion.set(qid, byQuestion.has(qid) ? null : { index, values: answerValues(answer) });
    }
  });
  evidence.answers = recorded;
}

function observedOptions(opts: LongPredictOptions, trackAnswers: boolean): [PredictOptions, Evidence] {
  const evidence: Evidence = { answered: false, states: null, questionTypes: null, questions: null,
    maxLen: null, headMaxLen: null, answers: null, resultCount: null, trackAnswers };
  const hooks = normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd);
  hooks.push({ onPredictStart(ctx) {
    evidence.answered = ctx.results !== null;
    evidence.states = Array.isArray(ctx.states) ? [...ctx.states] : null;
    if (!evidence.answered && isDict(ctx.questions)) {
      evidence.questionTypes = Object.fromEntries(Object.entries(ctx.questions).map(([qid, q]) => [qid, isDict(q) ? q.type : null]));
    }
    evidence.maxLen = ctx.maxLen;
    evidence.headMaxLen = ctx.headMaxLen;
    evidence.questions = isDict(ctx.questions) ? { ...ctx.questions } : null;
  } });
  const call: PredictOptions = { lang: opts.lang, hooks, hooksRaise: opts.hooksRaise, hooksTimeout: opts.hooksTimeout };
  if (defaultsAlreadyRan(opts)) markDefaultsRan(call);
  recorders.set(call, evidence);
  return [call, evidence];
}

function stateRoom(agent: Agent, questions: Record<string, QuestionDef>, maxLen: number, headMaxLen: number): number[] {
  return Object.values(questions).map((q) => Math.max(0, maxLen - buildQuestionPrefix(agent.tok, toInternal(q), maxLen, headMaxLen).ids.length - 1));
}

function windowBudget(agent: Agent, questions: Record<string, QuestionDef>, opts: LongPredictOptions): [number, number, number] {
  const requested = opts.window ?? Math.max(64, agent.maxLen - agent.headMaxLen - 8);
  const rooms = stateRoom(agent, questions, agent.maxLen, agent.headMaxLen);
  const room = rooms.length ? Math.min(...rooms) : requested;
  if (room <= 0) throw new Error(`predictLong: the questions' options fill the whole sequence (maxLen=${agent.maxLen}, headMaxLen=${agent.headMaxLen}), leaving no room for the state`);
  const size = Math.min(requested, room);
  if (size < requested && (opts.window != null || requested >= room * 2)) {
    console.warn(`laya: predictLong: questions leave ${room} state tokens inside maxLen=${agent.maxLen}; capping window ${requested} -> ${size}. Fewer or shorter options reduce the number of full forward passes.`);
  }
  let step = opts.stride ?? Math.max(1, Math.floor(size / 2));
  if (step > size) {
    if (size < requested && step <= requested) {
      const adjusted = Math.max(1, Math.floor(size / 2));
      console.warn(`laya: predictLong: window was reduced to ${size}; adjusting stride ${step} -> ${adjusted} to avoid unread gaps`);
      step = adjusted;
    } else {
      throw new Error(`predictLong: stride=${step} steps past the ${size}-token window; pass stride <= ${size}`);
    }
  }
  return [size, step, room];
}

function checkScanBudget(agent: Agent, evidence: Evidence, sized: number, asked: Record<string, QuestionDef>): void {
  if (evidence.answered || !evidence.questions || !Object.keys(evidence.questions).length) return;
  const maxLen = evidence.maxLen ?? agent.maxLen;
  const headMaxLen = evidence.headMaxLen ?? agent.headMaxLen;
  if (maxLen === agent.maxLen && headMaxLen === agent.headMaxLen && sameValue(evidence.questions, asked)) return;
  let rooms: number[];
  try { rooms = stateRoom(agent, evidence.questions as Record<string, QuestionDef>, maxLen, headMaxLen); }
  catch { return; } // Malformed questions belong to the normal inference validator.
  const room = Math.min(...rooms);
  if (room < sized) throw new Error(`predictLong: after the start hooks ran, the questions and token budget (maxLen=${maxLen} headMaxLen=${headMaxLen}) leave ${room} state tokens per window, but the scan was sized for ${sized} from the agent's config (maxLen=${agent.maxLen} headMaxLen=${agent.headMaxLen}). Every window would be re-truncated and parts of the document would reach no model. Pass window/stride explicitly, or call predictBatch directly, if a start hook changes either.`);
}

export async function predictLong(agent: Agent, state: unknown, questions: Record<string, QuestionDef>, opts: LongPredictOptions): Promise<LongResult> {
  if (state === null || state === undefined) throw new TypeError("state must not be None; pass a string, dict, or list");
  if (!isDict(questions)) throw new TypeError("questions must be a dict of question id -> definition");
  validateLongOptions(opts);
  for (const [qid, q] of Object.entries(questions)) checkQuestion(qid, q);
  const [budget, step, room] = windowBudget(agent, questions, opts);
  const asked = copyValue(questions);
  const stateIds = agent.tok.encode(serializeState(state).split(agent.tok.maskToken).join(" "));
  if (stateIds.length <= (opts.window != null ? budget : room)) {
    const [call, evidence] = observedOptions(opts, false);
    let single: unknown;
    try { single = await agent.systemOne(state, questions, call); }
    finally { recorders.delete(call); }
    // Correctness checks are outside the hook dispatcher and its suppression/timeout policy.
    // Inference has already run; changing that order would change the hook contract.
    checkScanBudget(agent, evidence, Math.max(budget, stateIds.length), asked);
    if (evidence.answered) {
      if (evidence.resultCount !== 1) throw new Error(`predictLong: a start hook answered this state with ${evidence.resultCount} results; ctx.skip() takes one result for the document`);
      console.warn("laya: predictLong: a hook answered the state before it was scanned, so no window decided the result");
      return skippedLongResult(single);
    }
    validateLongResult(single, "predictLong result", false);
    return { ...single, usage: { ...single.usage, windows: 1 } };
  }
  if (typeof agent.tok.decode !== "function") {
    const name = (agent.tok as { name?: unknown }).name ?? agent.tok.constructor?.name ?? "custom tokenizer";
    throw new Error(`predictLong: tokenizer ${JSON.stringify(name)} needs decode(ids) to split a long state`);
  }
  const windows: string[] = [], starts: number[] = [];
  for (let start = 0; start < stateIds.length; start += step) {
    windows.push(agent.tok.decode(stateIds.slice(start, start + budget)));
    starts.push(start);
    if (start + budget >= stateIds.length) break;
  }
  const [call, evidence] = observedOptions(opts, true);
  const configBudget = Math.max(64, agent.maxLen - agent.headMaxLen - 8);
  const unclamped = Math.max(1, Math.ceil(windows.length * Math.max(1, Math.floor(budget / 2)) / Math.max(1, Math.floor(configBudget / 2))));
  call.batchSize = opts.batchSize ?? (budget < configBudget && windows.length > unclamped * 2 ? unclamped : null);
  let results: unknown;
  try { results = await agent.predictBatch([...windows], questions, call); }
  finally { recorders.delete(call); }
  checkScanBudget(agent, evidence, budget, asked);
  if (!Array.isArray(results)) throw new TypeError("predictLong: prediction hooks must return an array of results");
  if (evidence.answered) {
    if (results.length !== 1) throw new Error(`predictLong: a start hook answered this state with ${results.length} results; ctx.skip() takes one result for the document, not one per window`);
    console.warn("laya: predictLong: a hook answered the state before it was scanned, so no window decided the result");
    return skippedLongResult(results[0]);
  }
  const rewritten = evidence.states !== null && !sameValue(evidence.states, windows);
  if (results.length !== windows.length && !rewritten) throw new Error(`predictLong: the state was split into ${windows.length} windows and the call returned ${results.length} results`);
  if (!results.length) {
    console.warn("laya: predictLong: a start hook left no states to score, so the call aggregated nothing and returns no answers");
    return { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0, windows: 0 } };
  }
  for (const result of results) validateLongResult(result, "predictLong window result", false);
  const rows = results as LongResult[];
  const questionTypes = evidence.questionTypes ?? Object.fromEntries(Object.entries(questions).map(([qid, q]) => [qid, q.type]));
  const answers: Record<string, LongAnswer> = {};
  for (const [qid, qtype] of Object.entries(questionTypes)) {
    const missing = rows.filter((r) => !Object.hasOwn(r.answers, qid)).length;
    if (missing) {
      if (missing === rows.length && !Object.hasOwn(asked, qid) && evidence.questionTypes !== null) continue;
      throw new Error(`predictLong: answer ${JSON.stringify(qid)} is missing from ${missing} of ${rows.length} windows after prediction hooks`);
    }
    const per = rows.map((r) => r.answers[qid]);
    if (per.some((answer) => answer.type !== per[0].type)) throw new Error(`predictLong: answer ${JSON.stringify(qid)} has inconsistent types across windows`);
    // The inference schema chooses the aggregation rule, even when end hooks uniformly
    // rewrite the answer schema. Such edits invalidate attribution, not the result itself.
    const metric = (answer: LongAnswer): number => qtype === "noul" ? (answer as { noul: number }).noul : answer.answer_confidence;
    if (per.some((answer) => !Number.isFinite(metric(answer)))) throw new TypeError(`predictLong: answer ${JSON.stringify(qid)} is missing the numeric aggregation field for ${String(qtype)}`);
    let best = 0;
    for (let i = 1; i < per.length; i++) if (metric(per[i]) > metric(per[best])) best = i;
    const original = per[best];
    let index: number | null = null;
    if (!rewritten) {
      if (evidence.answers === null) index = best; // A replacement predictBatch did not dispatch hooks.
      else {
        const record = evidence.answers.get(original)?.get(qid);
        if (record && sameValue(record.values, answerValues(original as unknown as Dict))) index = record.index;
      }
    }
    const answer = { ...original };
    delete answer.window;
    if (index !== null) answer.window = { index, token_start: starts[index], token_end: Math.min(starts[index] + budget, stateIds.length), count: rows.length };
    Object.defineProperty(answers, qid, { value: answer, enumerable: true, writable: true, configurable: true });
  }
  const usage: Dict = Object.create(null);
  for (const row of rows) {
    for (const [key, value] of Object.entries(row.usage)) {
      const previous = usage[key];
      if (typeof value === "number" || typeof value === "boolean") {
        usage[key] = (typeof previous === "number" || typeof previous === "boolean" ? Number(previous) : 0) + Number(value);
      } else if (isDict(value) && isDict(previous)) usage[key] = { ...previous, ...value };
      else Object.defineProperty(usage, key, { value, enumerable: true, writable: true, configurable: true });
    }
  }
  const result = { model: "laya-rl-agent", answers, usage: { ...usage, output_tokens: 0, windows: rows.length } as LongUsage };
  // Individually finite window counters can still overflow when summed. The Agent
  // must uphold the same result contract that the Router checks at its boundary.
  validateLongResult(result);
  return result;
}
