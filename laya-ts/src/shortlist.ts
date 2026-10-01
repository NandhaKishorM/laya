import { maxOf, renderOptions, serializeState } from "./common.js";
import type { Batch } from "./providers.js";
import type { QuestionDef, SystemOneResult } from "./agent.js";

export const DEFAULT_SHORTLIST_K = 20;

export type EmbedFn = (texts: string[]) => Promise<number[][]> | number[][];

export interface ShortlistMeta {
  labels: string[];
  scores: number[] | null;
  k: number;
  n: number;
  passthrough: boolean;
}

function checkK(k: unknown): number {
  if (typeof k !== "number" || !Number.isInteger(k) || k < 1) {
    throw new Error(`k must be a positive integer, got ${JSON.stringify(k)}`);
  }
  return k;
}

function criteriaItems(criteria: unknown): [string, unknown][] {
  let items: [string, unknown][];
  if (Array.isArray(criteria)) {
    items = (criteria as unknown[]).map((item) => [String(item), null] as [string, unknown]);
  } else if (typeof criteria === "object" && criteria !== null) {
    items = Object.entries(criteria as Record<string, unknown>);
  } else {
    throw new TypeError(
      `choice criteria must be a dict or list, got ${Array.isArray(criteria) ? "list" : criteria === null ? "NoneType" : typeof criteria}`,
    );
  }
  if (items.length === 0) throw new Error("choice criteria must contain at least one option");
  const seen = new Set<string>();
  for (const [key] of items) {
    if (seen.has(key)) throw new Error(`choice criteria label ${JSON.stringify(key)} is duplicated`);
    seen.add(key);
  }
  return items;
}

function optionTexts(items: [string, unknown][]): string[] {
  const crit: Record<string, unknown> = Object.fromEntries(items);
  const rendered = renderOptions({ t: "choice", ins: "", crit });
  if (rendered.length !== items.length) throw new Error("could not render every choice option");
  return rendered.map((p) => (typeof p === "string" ? p : String(p)));
}

function queryText(state: unknown, instructions?: unknown): string {
  const body = serializeState(state);
  if (instructions === null || instructions === undefined || instructions === "") return body;
  if (typeof instructions !== "string") return `${JSON.stringify(instructions)}\n${body}`;
  return `${instructions}\n${body}`;
}

function clean(v: number): number {
  return typeof v !== "number" || !Number.isFinite(v) ? 0 : v;
}

async function embeddings(embedFn: unknown, texts: string[]): Promise<number[][]> {
  if (typeof embedFn !== "function") throw new TypeError("embed_fn must be callable");
  const raw = await (embedFn as EmbedFn)([...texts]);
  const arr = raw as unknown;
  if (!Array.isArray(arr) || (arr as unknown[]).length !== texts.length) {
    throw new Error(`embed_fn must return an array of shape (${texts.length}, dim), got ${Array.isArray(arr) ? `(${(arr as unknown[]).length}, ?)` : typeof arr}`);
  }
  const rows = (arr as unknown[]).map((row) => {
    if (!Array.isArray(row) || row.length < 1) {
      throw new Error(`embed_fn must return an array of shape (${texts.length}, dim), got row ${JSON.stringify(row)?.slice(0, 60)}`);
    }
    return (row as unknown[]).map((v) => clean(Number(v)));
  });
  return rows;
}

function cosine(query: number[], docs: number[][]): number[] {
  const qn = Math.sqrt(query.reduce((a, v) => a + v * v, 0));
  if (qn === 0 || docs.length === 0) return docs.map(() => 0);
  return docs.map((d) => {
    const dn = Math.sqrt(d.reduce((a, v) => a + v * v, 0));
    if (dn === 0) return 0;
    const dot = d.reduce((a, v, i) => a + v * (query[i] ?? 0), 0);
    return Math.min(1, Math.max(-1, dot / (dn * qn)));
  });
}

async function rank(
  state: unknown,
  criteria: unknown,
  embedFn: EmbedFn,
  k: number,
  instructions?: unknown,
): Promise<{ labels: string[]; scores: number[] | null; passthrough: boolean; n: number }> {
  const checked = checkK(k);
  const items = criteriaItems(criteria);
  const n = items.length;
  const keys = items.map(([key]) => key);
  if (checked >= n) return { labels: [...keys], scores: null, passthrough: true, n };
  const q = queryText(state, instructions);
  const matrix = await embeddings(embedFn, [q, ...optionTexts(items)]);
  const sims = cosine(matrix[0], matrix.slice(1));
  const order = sims
    .map((s, i) => i)
    .sort((a, b) => sims[b] - sims[a]);
  const top = order.slice(0, checked);
  return { labels: top.map((i) => keys[i]), scores: top.map((i) => sims[i]), passthrough: false, n };
}

function subsetCriteria(criteria: unknown, labels: string[]): unknown {
  if (Array.isArray(criteria)) return [...labels];
  return Object.fromEntries(labels.map((l) => [l, (criteria as Record<string, unknown>)[l]]));
}

/** Return the top-k choice labels for state. Skips embed_fn when k >= n. */
export async function shortlistChoice(
  state: unknown,
  criteria: unknown,
  embedFn: EmbedFn,
  k: number = DEFAULT_SHORTLIST_K,
  instructions?: unknown,
): Promise<string[]> {
  const { labels } = await rank(state, criteria, embedFn, k, instructions);
  return labels;
}

/** Shortlist each choice question, then call predict/systemOne once. */
export async function predictShortlist(
  agent: unknown,
  state: unknown,
  questions: Record<string, QuestionDef>,
  embedFn?: EmbedFn | null,
  k: number = DEFAULT_SHORTLIST_K,
  predictKwargs: Record<string, unknown> = {},
): Promise<SystemOneResult & { shortlist: Record<string, ShortlistMeta> }> {
  if (typeof questions !== "object" || questions === null || Array.isArray(questions)) {
    throw new TypeError("questions must be a dict of question id -> definition");
  }
  const checked = checkK(k);
  const needsEmbed = Object.values(questions).some(
    (qdef) =>
      typeof qdef === "object" && qdef !== null && !Array.isArray(qdef) &&
      (qdef as Record<string, unknown>)["type"] === "choice" &&
      "criteria" in (qdef as Record<string, unknown>) &&
      Object.keys((qdef as Record<string, unknown>)["criteria"] as object ?? {}).length > checked,
  );
  // Default bi-encoder is the agent itself; pass a dedicated embedFn only when it shortlists better.
  const embed = embedFn ?? (needsEmbed ? embedFnFromAgent(agent) : null);
  const reduced: Record<string, unknown> = {};
  const meta: Record<string, ShortlistMeta> = {};
  for (const [qid, qdef] of Object.entries(questions)) {
    if (typeof qdef !== "object" || qdef === null || Array.isArray(qdef) || (qdef as Record<string, unknown>)["type"] !== "choice") {
      reduced[qid] = qdef;
      continue;
    }
    if (!("criteria" in (qdef as Record<string, unknown>))) {
      throw new Error(`question ${JSON.stringify(qid)} is a choice but has no criteria`);
    }
    const q = qdef as Record<string, unknown>;
    const { labels, scores, passthrough, n } = await rank(state, q["criteria"], embed as EmbedFn, checked, q["instructions"]);
    meta[qid] = { labels: [...labels], scores, k: checked, n, passthrough };
    if (passthrough) {
      reduced[qid] = qdef;
      continue;
    }
    reduced[qid] = { ...q, criteria: subsetCriteria(q["criteria"], labels) };
  }
  const a = agent as Record<string, unknown>;
  const fn = a?.["predict"] ?? a?.["systemOne"] ?? a?.["system_one"];
  if (typeof fn !== "function") throw new TypeError("agent must provide predict or system_one");
  const extra = Object.keys(predictKwargs ?? {}).length ? [predictKwargs] : [];
  const result = await (fn as (...args: unknown[]) => Promise<unknown>).call(agent, state, reduced, ...extra);
  if (typeof result !== "object" || result === null || Array.isArray(result)) {
    throw new TypeError("predict/system_one must return a dict");
  }
  return { ...(result as SystemOneResult), shortlist: meta };
}

/** Mean-pool the agent's provider encoder. Throws honestly without encoder access. */
export function embedFnFromAgent(agent: unknown, maxLength = 512, batchSize = 32): EmbedFn {
  if (!Number.isInteger(maxLength) || (maxLength as number) < 1) {
    throw new Error(`max_length must be a positive integer, got ${JSON.stringify(maxLength)}`);
  }
  if (!Number.isInteger(batchSize) || (batchSize as number) < 1) {
    throw new Error(`batch_size must be a positive integer, got ${JSON.stringify(batchSize)}`);
  }
  const a = agent as { tok?: { encode(t: string): number[]; padId: number }; provider?: { runEncoder(b: Batch): Promise<{ lastHidden: number[][][] }> } };
  if (!a?.provider || typeof a.provider.runEncoder !== "function" || !a?.tok || typeof a.tok.encode !== "function") {
    throw new Error("embedFnFromAgent needs encoder access: agent.provider.runEncoder + agent.tok.encode required");
  }
  const tok = a.tok;
  const provider = a.provider;
  return async (texts: string[]): Promise<number[][]> => {
    const rows = (texts ?? []).map((t) => (t === null || t === undefined ? "" : String(t)));
    if (rows.length === 0) return [];
    const out: number[][] = [];
    for (let s = 0; s < rows.length; s += batchSize) {
      const chunk = rows.slice(s, s + batchSize);
      const ids = chunk.map((t) => tok.encode(t).slice(0, maxLength));
      const L = maxOf(ids.map((r) => r.length), 1);
      const pad = tok.padId ?? 0;
      const batch: Batch = {
        inputIds: ids.map((r) => [...r, ...Array(L - r.length).fill(pad)]),
        attentionMask: ids.map((r) => [...Array(r.length).fill(1), ...Array(L - r.length).fill(0)]),
        markerPos: ids.map(() => [0]),
        markerMask: ids.map(() => [true]),
        qtype: ids.map(() => 0),
      };
      const { lastHidden } = await provider.runEncoder(batch);
      for (let i = 0; i < chunk.length; i++) {
        const mask = batch.attentionMask[i];
        const denom = Math.max(1, mask.reduce((x, y) => x + y, 0));
        const dim = lastHidden[i]?.[0] ? (lastHidden[i][0] as number[]).length : 0;
        const pooled = new Array(dim).fill(0);
        for (let j = 0; j < mask.length; j++) {
          if (!mask[j]) continue;
          const hv = lastHidden[i]?.[j] as number[] | undefined;
          if (!hv) continue;
          for (let h = 0; h < dim; h++) pooled[h] += Number(hv[h] ?? 0);
        }
        out.push(pooled.map((v) => v / denom));
      }
    }
    return out;
  };
}
