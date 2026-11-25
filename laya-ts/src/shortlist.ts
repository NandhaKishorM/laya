import { maxOf, renderOptions, serializeState } from "./common.js";
import type { Batch } from "./providers.js";
import type { QuestionDef, SystemOneResult } from "./agent.js";

export const DEFAULT_SHORTLIST_K = 20;
export const DEFAULT_TOURNAMENT_GROUP = 16;
export const DEFAULT_SHORTLIST_MAXSIZE = 4096;

export type EmbedFn = (texts: string[]) => Promise<number[][]> | number[][];

export interface ShortlistMeta {
  labels: string[];
  scores: number[] | null;
  k: number;
  n: number;
  passthrough: boolean;
}

export interface TournamentMeta {
  labels: string[];
  n: number;
  rounds: number;
}

export interface CacheInfo {
  size: number;
  maxsize: number;
  hits: number;
  misses: number;
}

export interface CachedEmbedFn {
  (texts: string[]): Promise<number[][]>;
  cacheInfo(): CacheInfo;
  cache_info(): CacheInfo;
  cacheClear(): void;
  cache_clear(): void;
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

/** Return `question` without an `option_order` that no longer fits it.
 *
 * `option_order` holds one slot per option, so an order that fitted the caller's label set does not
 * fit a narrowed one -- and once the agent reads the field at all, that mismatch is a refused
 * question, not a silently ignored one. Remapping the order onto the kept labels would move options
 * between slots, so it is dropped instead, once per question per call. Mirrors
 * `laya.shortlist._drop_stale_order` (`laya/shortlist.py:428`), which both `predict_shortlist` and
 * `predict_tournament` run on every narrowed choice; a passthrough keeps its order. */
function dropStaleOrder(qid: string, question: Record<string, unknown>, nOptions: number,
  warned: Set<string>): Record<string, unknown> {
  if (!Object.hasOwn(question, "option_order")) return question;
  const out = { ...question };
  const order = out["option_order"];
  delete out["option_order"];
  if (!warned.has(qid)) {
    warned.add(qid);
    // The same shape as the reference's warning: the slot count the order carried, so a caller can
    // see how much of the label set the narrowing dropped.
    const had = Array.isArray(order) ? `${order.length} slots` : JSON.stringify(order);
    console.warn(
      `laya.shortlist: question ${JSON.stringify(qid)}: 'option_order' (${had}) was dropped because the ` +
        `label set was narrowed to ${nOptions}; the order no longer fits the kept options and remapping ` +
        `it onto them would move options between slots.`,
    );
  }
  return out;
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
  const warned = new Set<string>();
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
    reduced[qid] = dropStaleOrder(
      qid, { ...q, criteria: subsetCriteria(q["criteria"], labels) }, labels.length, warned);
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

/**
 * Reduce large choices through balanced elimination rounds, then predict all questions.
 * Each round batches every group's question in one call. Metadata lists the finalists;
 * probabilities, confidence, usage and other result fields come from the final call only.
 */
export async function predictTournament(
  agent: unknown,
  state: unknown,
  questions: Record<string, QuestionDef>,
  groupSize: number = DEFAULT_TOURNAMENT_GROUP,
  predictKwargs: Record<string, unknown> = {},
): Promise<SystemOneResult & { tournament: Record<string, TournamentMeta> }> {
  if (typeof questions !== "object" || questions === null || Array.isArray(questions) ||
      (Object.getPrototypeOf(questions) !== null && Object.getPrototypeOf(questions) !== Object.prototype)) {
    throw new TypeError("questions must be a dict of question id -> definition");
  }
  if (!Number.isSafeInteger(groupSize) || groupSize < 2) {
    throw new Error(`groupSize must be a safe integer >= 2, got ${JSON.stringify(groupSize)}`);
  }
  const meta = new Map<string, TournamentMeta>();
  for (const [qid, qdef] of Object.entries(questions)) {
    if (typeof qdef !== "object" || qdef === null || Array.isArray(qdef) || qdef["type"] !== "choice") continue;
    if (!("criteria" in qdef)) {
      throw new Error(`question ${JSON.stringify(qid)} is a choice but has no criteria`);
    }
    const criteria = qdef["criteria"];
    if (Array.isArray(criteria)) {
      for (const label of criteria) {
        if (label === null || label === undefined || typeof label === "object") {
          throw new Error(`question ${JSON.stringify(qid)}: choice labels must not be null, lists or dicts`);
        }
      }
    }
    const labels = criteriaItems(criteria).map(([label]) => label);
    meta.set(qid, { labels, n: labels.length, rounds: 0 });
  }
  const a = agent as Record<string, unknown>;
  const fn = a?.["predict"] ?? a?.["systemOne"] ?? a?.["system_one"];
  if (typeof fn !== "function") throw new TypeError("agent must provide predict or system_one");
  const extra = Object.keys(predictKwargs ?? {}).length ? [predictKwargs] : [];
  const predict = async (current: Record<string, QuestionDef>): Promise<SystemOneResult> => {
    const result = await (fn as (...args: unknown[]) => unknown).call(agent, state, current, ...extra);
    if (typeof result !== "object" || result === null || Array.isArray(result)) {
      throw new TypeError("predict/system_one must return a dict");
    }
    return result as SystemOneResult;
  };
  const warned = new Set<string>();
  const cut = (qid: string, labels: string[]): QuestionDef => dropStaleOrder(
    qid,
    { ...questions[qid], criteria: subsetCriteria(questions[qid]["criteria"], labels) },
    labels.length,
    warned,
  ) as QuestionDef;

  while (true) {
    const groups: { qid: string; labels: string[] }[] = [];
    for (const [qid, entry] of meta) {
      const n = entry.labels.length;
      const parts = Math.ceil(n / groupSize);
      if (parts <= 1) continue;
      for (let i = 0; i < parts; i++) {
        groups.push({ qid, labels: entry.labels.slice(Math.floor(i * n / parts), Math.floor((i + 1) * n / parts)) });
      }
    }
    if (groups.length === 0) break;
    const roundQuestions = Object.fromEntries(groups.map(({ qid, labels }, i) => [String(i), cut(qid, labels)]));
    const result = await predict(roundQuestions);
    const answers = result["answers"];
    if (typeof answers !== "object" || answers === null || Array.isArray(answers)) {
      throw new TypeError("tournament round must return an answers dict");
    }
    const winners = new Map<string, string[]>();
    for (const [i, { qid, labels }] of groups.entries()) {
      const answer = Object.hasOwn(answers, String(i)) ? (answers as Record<string, unknown>)[String(i)] : undefined;
      const choice = typeof answer === "object" && answer !== null && !Array.isArray(answer) && Object.hasOwn(answer, "choice")
        ? (answer as Record<string, unknown>)["choice"] : undefined;
      if (typeof choice !== "string" || !labels.includes(choice)) {
        throw new Error(`tournament group ${i} for question ${JSON.stringify(qid)} must return a choice from its criteria`);
      }
      const kept = winners.get(qid) ?? [];
      kept.push(choice);
      winners.set(qid, kept);
    }
    for (const [qid, labels] of winners) {
      const entry = meta.get(qid)!;
      entry.labels = labels;
      entry.rounds++;
    }
  }

  const finalQuestions = Object.fromEntries(Object.entries(questions).map(([qid, qdef]) => {
    const entry = meta.get(qid);
    return [qid, entry?.rounds ? cut(qid, entry.labels) : qdef];
  }));
  const result = await predict(finalQuestions);
  return { ...result, tournament: Object.fromEntries(meta) };
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

/**
 * Wrap `embedFn` with an LRU cache keyed by text.
 *
 * Shortlisting embeds the query plus every option on every call. When the option
 * set is fixed -- a label list, an intent taxonomy -- those rows never change
 * between requests and are re-embedded every time. Wrapping the embedder once
 * leaves the first call unchanged and reduces each repeat to embedding the new
 * query alone.
 *
 * A cold cache costs the same number of embedder calls as the unwrapped function:
 * the misses are deduplicated and embedded in one call. Nothing is cached when
 * the embedder throws or returns a bad shape, so a transient failure cannot
 * poison the cache.
 *
 * @param embedFn The underlying embedding function
 * @param maxsize Maximum number of distinct text rows to cache (default: 4096)
 */
export function cachedEmbedFn(embedFn: EmbedFn, maxsize: number = DEFAULT_SHORTLIST_MAXSIZE): CachedEmbedFn {
  if (typeof embedFn !== "function") {
    throw new TypeError("embed_fn must be callable");
  }
  if (typeof maxsize !== "number" || !Number.isInteger(maxsize) || maxsize < 1) {
    throw new Error(`maxsize must be a positive integer, got ${JSON.stringify(maxsize)}`);
  }

  const rowsByText = new Map<string, number[]>();
  let hitsTotal = 0;
  let missesTotal = 0;
  let generation = 0;

  const fn = async (texts: string[]): Promise<number[][]> => {
    const keys = (texts ?? []).map((t) => (t === null || t === undefined ? "" : String(t)));
    if (keys.length === 0) return [];

    const found = new Map<string, number[]>();
    let hitsThisCall = 0;

    for (const key of keys) {
      const row = rowsByText.get(key);
      if (row !== undefined) {
        found.set(key, row);
        hitsThisCall++;
      }
    }

    hitsTotal += hitsThisCall;
    missesTotal += keys.length - hitsThisCall;

    const missing: string[] = [];
    const seenMissing = new Set<string>();
    for (const key of keys) {
      if (!found.has(key) && !seenMissing.has(key)) {
        seenMissing.add(key);
        missing.push(key);
      }
    }

    const genAtStart = generation;

    if (missing.length > 0) {
      const fresh = await embeddings(embedFn, missing);

      // Validate dimensions of all returned rows before caching any
      const heldDim = rowsByText.size > 0 ? rowsByText.values().next().value?.length : undefined;
      const expectedDim = heldDim ?? (fresh.length > 0 ? fresh[0]?.length : undefined);

      for (let i = 0; i < fresh.length; i++) {
        const rowDim = fresh[i]?.length ?? 0;
        if (expectedDim !== undefined && rowDim !== expectedDim) {
          if (heldDim !== undefined) {
            throw new Error(
              `the embedder returned dim ${rowDim}, but the cache holds dim ${heldDim}; call cacheClear() if the model behind it changed`,
            );
          } else {
            throw new Error(
              `the embedder returned inconsistent embedding dimensions (${expectedDim} vs ${rowDim})`,
            );
          }
        }
      }

      for (let i = 0; i < missing.length; i++) {
        found.set(missing[i], fresh[i]);
      }

      // If cacheClear() ran while embeddings() was pending, do not restore entries into the cleared cache
      if (generation !== genAtStart) {
        return keys.map((key) => [...(found.get(key) ?? [])]);
      }
    }

    // Apply recency updates and evictions in input key order
    for (const key of keys) {
      const row = found.get(key);
      if (row !== undefined) {
        rowsByText.delete(key);
        rowsByText.set(key, row);
        while (rowsByText.size > maxsize) {
          const victim = rowsByText.keys().next().value!;
          rowsByText.delete(victim);
        }
      }
    }

    return keys.map((key) => [...(found.get(key) ?? [])]);
  };

  const cacheInfo = (): CacheInfo => ({
    size: rowsByText.size,
    maxsize,
    hits: hitsTotal,
    misses: missesTotal,
  });

  const cacheClear = (): void => {
    rowsByText.clear();
    hitsTotal = 0;
    missesTotal = 0;
    generation++;
  };

  const cachedResult = fn as CachedEmbedFn;
  cachedResult.cacheInfo = cacheInfo;
  cachedResult.cache_info = cacheInfo;
  cachedResult.cacheClear = cacheClear;
  cachedResult.cache_clear = cacheClear;

  return cachedResult;
}

export const cached = cachedEmbedFn;

