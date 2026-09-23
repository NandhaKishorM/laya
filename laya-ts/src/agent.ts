import {
  TEMP_MAX,
  TEMP_MIN,
  buildSequence,
  clampTemperature,
  collateItems,
  confidenceFromProbs,
  renderOptions,
  softmax,
  tempBucket,
} from "./common.js";
import type { Batch, SessionProvider } from "./providers.js";
import { encodeWithData, parseTokenizerJson, type TokenizerLike } from "./tokenizer.js";

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
  action: ActionInfo;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, unknown>;
  probabilities: Record<string, number>;
  confidence: number;
  action: ActionInfo;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
  confidence: number;
  action: ActionInfo;
}

export type SystemAnswer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface SystemUsage {
  input_tokens: number;
  output_tokens: number;
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
  max_len?: number;
  head_max_len?: number;
  temperature?: unknown;
  temperature_by_options?: Record<string, unknown>;
}

function qidStr(qid: string): string {
  return JSON.stringify(qid);
}

export function checkQuestion(qid: string, qdef: unknown): void {
  if (typeof qdef !== "object" || qdef === null || Array.isArray(qdef)) {
    const got = Array.isArray(qdef) ? "list" : qdef === null ? "NoneType" : typeof qdef;
    throw new Error(`question ${qidStr(qid)}: definition must be a dict, got ${got}`);
  }
  const q = qdef as Record<string, unknown>;
  const t = q["type"];
  if (t !== "choice" && t !== "score" && t !== "noul") {
    throw new Error(
      `question ${qidStr(qid)}: unknown type ${JSON.stringify(t)}; use one of ${JSON.stringify(sorted(QTYPES))}`,
    );
  }
  if (!("instructions" in q)) {
    throw new Error(`question ${qidStr(qid)}: no 'instructions'; add the text the model should answer`);
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
  } else if (t === "score") {
    if (!Array.isArray(crit)) {
      throw new Error(
        `question ${qidStr(qid)}: a score question takes 'criteria' as a list of level descriptions, index 0 first`,
      );
    }
    if (crit.length === 0) {
      throw new Error(`question ${qidStr(qid)}: a score question needs at least one level`);
    }
  } else if (crit !== undefined && crit !== null && (typeof crit !== "object" || Array.isArray(crit))) {
    throw new Error(
      `question ${qidStr(qid)}: a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it`,
    );
  }
}

function sorted(o: Record<string, number>): string[] {
  return Object.keys(o).sort();
}

export function toInternal(qdef: QuestionDef): { t: "choice" | "score" | "noul"; ins: string; crit: unknown } {
  const t = qdef["type"] as "choice" | "score" | "noul";
  let crit: unknown = qdef["criteria"];
  if (t === "choice" && Array.isArray(crit)) {
    crit = Object.fromEntries(crit.map((c: unknown) => [c as string, null]));
  } else if (t === "noul" && crit !== null && crit !== undefined && typeof crit === "object" && !Array.isArray(crit)) {
    crit = Object.fromEntries(Object.entries(crit as Record<string, unknown>).map(([k, v]) => [String(k).toLowerCase(), v]));
  }
  let ins: unknown = qdef["instructions"];
  if (typeof ins !== "string") ins = JSON.stringify(ins);
  return { t, ins: ins as string, crit };
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

/** Fallback when tokenizer.json is absent: ids from rl_agent_config, if present. */
function tokenizerFromConfig(cfg: AgentCfg): TokenizerLike | null {
  const c = cfg as Record<string, unknown>;
  const num = (v: unknown): number | null =>
    typeof v === "number" && Number.isFinite(v) ? v : null;
  const cls = num(c["cls_token_id"]) ?? num(c["cls_id"]);
  const sep = num(c["sep_token_id"]) ?? num(c["sep_id"]);
  const mask = num(c["mask_token_id"]) ?? num(c["mask_id"]);
  const pad = num(c["pad_token_id"]) ?? num(c["pad_id"]);
  if (cls === null && sep === null && mask === null && pad === null) return null;
  const d = defaultTokenizer();
  return {
    clsId: cls ?? d.clsId,
    sepId: sep ?? d.sepId,
    maskId: mask ?? d.maskId,
    padId: pad ?? d.padId,
    maskToken: typeof c["mask_token"] === "string" ? (c["mask_token"] as string) : d.maskToken,
    encode: d.encode,
  };
}

const r4 = (v: number): number => Math.round(v * 1e4) / 1e4;

export class Agent {
  cfg: AgentCfg;
  provider: SessionProvider;
  tok: TokenizerLike;
  maxLen: number;
  headMaxLen: number;
  temperatureRaw: unknown;
  temperatureByOptionsRaw: Record<string, unknown>;
  temperature: number[];
  temperatureByOptions: Record<string, number>;

  constructor(opts: AgentOptions) {
    if (!opts || !opts.provider) throw new Error("Agent needs a provider");
    this.provider = opts.provider;
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
    const rawList = Array.isArray(raw) ? raw : [raw, raw, raw];
    this.temperature = [0, 1, 2].map((i) => clampTemperature(rawList[i] ?? 1.0));
    this.temperatureByOptionsRaw = (cfg.temperature_by_options ?? {}) as Record<string, unknown>;
    this.temperatureByOptions = Object.fromEntries(
      Object.entries(this.temperatureByOptionsRaw).map(([k, v]) => [k, clampTemperature(v)]),
    );
    const entries: Array<[string, unknown, number]> = [
      ...Object.entries(this.temperatureByOptionsRaw).map(
        ([k, v]) => [k, v, this.temperatureByOptions[k]] as [string, unknown, number],
      ),
      ...[0, 1, 2].map(
        (i) => [`temperature[${i}]`, rawList[i] ?? 1.0, this.temperature[i]] as [string, unknown, number],
      ),
    ];
    const rejected: string[] = [];
    for (const [name, rawV, applied] of entries) {
      try {
        if (Number(rawV) === applied) continue;
      } catch {
        /* invalid entries already have a neutral fallback; still report */
      }
      rejected.push(`${name}=${JSON.stringify(rawV) ?? String(rawV)} -> ${applied}`);
    }
    if (rejected.length > 0) {
      console.warn(
        `laya: this checkpoint ships invalid temperatures or values outside [${TEMP_MIN}, ${TEMP_MAX}]; ` +
          `using ${rejected.join(", ")}. Treat confidence from the affected entries as uncalibrated.`,
      );
    }
  }

  async systemOne(state: unknown, questions: Record<string, QuestionDef>): Promise<SystemOneResult> {
    const ids = Object.keys(questions ?? {});
    if (ids.length === 0) {
      return { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    }
    const items: { ids: number[]; markers: number[]; qtype: number }[] = [];
    const internals: { t: "choice" | "score" | "noul"; ins: string; crit: unknown }[] = [];
    for (const qid of ids) {
      checkQuestion(qid, questions[qid]);
      const q = toInternal(questions[qid]);
      internals.push(q);
      const { ids: seq, markers } = buildSequence(this.tok, state, q, this.maxLen, this.headMaxLen);
      if (markers.length !== renderOptions(q).length) {
        throw new Error(`question ${qidStr(qid)} options exceed head_max_len=${this.headMaxLen}`);
      }
      items.push({ ids: seq, markers, qtype: QTYPES[q.t] });
    }
    const collated = collateItems([items], this.tok.padId);
    if (!collated) throw new Error("no items to collate");
    const batch: Batch = collated;
    const nTokens = batch.attentionMask.flat().reduce((a, b) => a + b, 0);
    const { lastHidden } = await this.provider.runEncoder(batch);
    const { logits, act } = await this.provider.runHead(lastHidden, batch);

    const answers: Record<string, SystemAnswer> = {};
    for (let r = 0; r < ids.length; r++) {
      const qid = ids[r];
      const q = internals[r];
      const k = items[r].markers.length;
      const qt = QTYPES[q.t];
      const bucket = tempBucket(qt, k);
      const scale = this.temperatureByOptions[bucket] ?? this.temperature[qt] ?? 1.0;
      const z = (logits[r] as number[]).slice(0, k).map((v) => v / scale);
      const p = softmax(z);
      const actRow = (act[r] as number[]) ?? [1, 0];
      const actP = softmax(actRow.slice(0, Math.max(2, actRow.length)));
      const ext = { act_probability: r4(actP[0]) };
      if (q.t === "choice") {
        const keys = Object.keys(q.crit as Record<string, unknown>);
        let best = 0;
        for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        answers[qid] = {
          type: "choice",
          choice: keys[best],
          probabilities: Object.fromEntries(keys.map((kk, i) => [kk, r4(p[i] ?? 0)])),
          confidence: r4(confidenceFromProbs(p)),
          action: ext,
        };
      } else if (q.t === "score") {
        const exp = p.reduce((a, v, i) => a + i * v, 0);
        answers[qid] = {
          type: "score",
          score: r4(exp),
          legend: Object.fromEntries((q.crit as unknown[]).map((c, i) => [String(i), c])),
          probabilities: Object.fromEntries(p.map((v, i) => [String(i), r4(v)])),
          confidence: r4(confidenceFromProbs(p)),
          action: ext,
        };
      } else {
        const pt = p[1] ?? 0;
        answers[qid] = {
          type: "noul",
          noul: r4(pt),
          confidence: r4(Math.max(pt, 1 - pt)),
          action: ext,
        };
      }
    }
    return { model: "laya-rl-agent", answers, usage: { input_tokens: nTokens, output_tokens: 0 } };
  }

  async predict(state: unknown, questions: Record<string, QuestionDef>): Promise<SystemOneResult> {
    return this.systemOne(state, questions);
  }

  static async load(
    modelDirOrRepo: string,
    opts?: {
      device?: string;
      subfolder?: string | null;
      localDir?: string;
      token?: string | null;
      numThreads?: number;
    },
  ): Promise<Agent> {
    const sub = opts?.subfolder ?? null;
    const isBrowser =
      typeof (globalThis as unknown as { window?: unknown }).window !== "undefined";
    let cfg: AgentCfg = {};
    let tokenizerJson: unknown | null = null;
    let dir = opts?.localDir ?? modelDirOrRepo;
    let provider: SessionProvider;
    if (isBrowser) {
      const { loadWebBundle, createWebProvider } = await import("./providers.js");
      const bundle = await loadWebBundle(modelDirOrRepo, { subfolder: sub });
      cfg = bundle.cfg;
      tokenizerJson = bundle.tokenizerJson;
      dir = bundle.dir;
      provider = await createWebProvider(dir, { numThreads: opts?.numThreads });
    } else {
      const { loadNodeBundle, createNodeProvider } = await import("./providers.js");
      const bundle = await loadNodeBundle(modelDirOrRepo, {
        subfolder: sub,
        localDir: opts?.localDir,
        token: opts?.token,
      });
      cfg = bundle.cfg;
      tokenizerJson = bundle.tokenizerJson;
      dir = bundle.dir;
      provider = await createNodeProvider(dir, { device: opts?.device, numThreads: opts?.numThreads });
    }
    let tok: TokenizerLike | null = null;
    try {
      tok = tokenizerJson ? tokenizerFromHF(tokenizerJson) : tokenizerFromConfig(cfg);
    } catch {
      tok = null;
    }
    return new Agent({ provider, tok: tok ?? defaultTokenizer(), cfg });
  }
}
