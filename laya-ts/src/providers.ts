/** ONNX session shim: Node (onnxruntime-node) + browser (onnxruntime-web).
 * Lazy imports only — unit tests with a fake provider never touch onnxruntime. */

export interface Batch {
  inputIds: number[][];
  attentionMask: number[][];
  markerPos: number[][];
  markerMask: boolean[][];
  qtype: number[];
}

export interface SessionProvider {
  runEncoder(batch: Batch): Promise<{ lastHidden: number[][][] }>;
  runHead(hidden: number[][][] | unknown, batch: Batch): Promise<{ logits: number[][]; act: number[][] }>;
}

function toNested(data: ArrayLike<number | bigint | boolean>, dims: number[]): any {
  // Single-pass copy + precomputed steps; no per-node slice/reduce.
  let total = 1;
  for (const d of dims) total *= d;
  const flat = new Array(total);
  for (let i = 0; i < total; i++) {
    const v = (data as any)[i];
    flat[i] = typeof v === "bigint" ? Number(v) : v;
  }
  if (dims.length === 0) return flat[0];
  const steps: number[] = new Array(dims.length);
  for (let d = 0; d < dims.length; d++) {
    let s = 1;
    for (let k = d + 1; k < dims.length; k++) s *= dims[k];
    steps[d] = s;
  }
  const rec = (d: number, off: number): any => {
    if (d === dims.length - 1) return flat.slice(off, off + dims[d]);
    const out: any[] = new Array(dims[d]);
    for (let i = 0; i < dims[d]; i++) out[i] = rec(d + 1, off + i * steps[d]);
    return out;
  };
  return rec(0, 0);
}

function i64(ort: any, arr: number[] | number[][], dims: number[]): any {
  // Rank <= 2 by construction; direct loop avoids flat(Infinity) intermediates.
  const out = new BigInt64Array(dims.reduce((a, b) => a * b, 1));
  let p = 0;
  if (Array.isArray((arr as any)[0])) {
    for (const row of arr as number[][]) for (const v of row) out[p++] = BigInt(Math.trunc(v));
  } else {
    for (const v of arr as number[]) out[p++] = BigInt(Math.trunc(v));
  }
  return new ort.Tensor("int64", out, dims);
}

/** Encoder feeds: input_ids + attention_mask (int64). */
export function feed(ort: any, b: Batch): Record<string, any> {
  const n = b.inputIds.length;
  let L = 1;
  for (const r of b.inputIds) if (r.length > L) L = r.length;
  return {
    input_ids: i64(ort, b.inputIds, [n, L]),
    attention_mask: i64(ort, b.attentionMask, [n, L]),
  };
}

/** Head feeds: encoder hidden + marker_pos/mask + qtype. */
export function feedHead(ort: any, hidden: number[][][] | any, b: Batch): Record<string, any> {
  const n = b.markerPos.length;
  let k = 1;
  for (const r of b.markerPos) if (r.length > k) k = r.length;
  // Direct flatten into typed arrays; no flat(Infinity)+map intermediates.
  const nH = (hidden as any).length ?? n;
  const S = (hidden as any)[0]?.length ?? 1;
  const Hd = (hidden as any)[0]?.[0]?.length ?? 1;
  const flatH = new Float32Array(nH * S * Hd);
  let p = 0;
  for (let i = 0; i < nH; i++) {
    const bi = (hidden as any)[i] ?? [];
    for (let j = 0; j < S; j++) {
      const hj = bi[j] ?? [];
      for (let h = 0; h < Hd; h++) flatH[p++] = Number(hj[h] ?? 0);
    }
  }
  const H = new ort.Tensor("float32", flatH, [nH, S, Hd]);
  // Pad/trim mask rows to S so the mask always matches hidden_states even
  // if a caller passes unpadded rows.
  const maskRows = b.attentionMask.map((r) => {
    const row = r.slice(0, S);
    while (row.length < S) row.push(0);
    return row;
  });
  return {
    hidden_states: H,
    marker_pos: i64(ort, b.markerPos, [n, k]),
    marker_mask: new ort.Tensor("bool", (() => {
      const out = new Uint8Array(n * k);
      let q = 0;
      for (const row of b.markerMask as unknown as boolean[][])
        for (let j = 0; j < k; j++) out[q++] = row[j] ? 1 : 0;
      return out;
    })(), [n, k]),
    qtype: i64(ort, b.qtype.map((v) => [v]), [n, 1]),
    // Padding mask for the head transformer (py DecisionModel.forward).
    // Without it, batch mates of unequal length corrupt each other's markers.
    attention_mask: i64(ort, maskRows, [n, S]),
  };
}

function pickOutput(out: Record<string, any>, names: string[]): any {
  for (const n of names) if (out[n] !== undefined) return out[n];
  const vals = Object.values(out);
  return vals[0];
}

export interface ProviderOptions {
  device?: string;
  numThreads?: number;
}

function applyNumThreads(ort: any, numThreads?: number): void {
  try {
    const raw =
      numThreads ??
      (typeof process !== "undefined" ? Number((process as any).env?.["LAYA_THREADS"]) : NaN);
    if (Number.isFinite(raw) && (raw as number) > 0 && ort?.env) {
      ort.env.numThreads = Math.trunc(raw as number);
    }
  } catch {
    /* best-effort only */
  }
}

function isOomError(e: unknown): boolean {
  const m = String((e as any)?.message ?? e).toLowerCase();
  return m.includes("memory") || m.includes("cuda") || m.includes("out of memory") || m.includes("oom");
}

/** Online-first fetch: try network, cache on success, fall back to CacheStorage. */
async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const g = globalThis as unknown as { caches?: any };
  let cache: any = null;
  let hit: any = null;
  try {
    if (g.caches && typeof g.caches.open === "function") {
      try {
        cache = await g.caches.open("laya-ts");
        try {
          hit = await cache.match(url);
        } catch {
          hit = null;
        }
      } catch {
        cache = null;
      }
    }
  } catch {
    cache = null;
  }
  if (cache) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        try {
          await cache.put(url, res.clone());
        } catch {
          /* cache full/blocked: still return network bytes */
        }
        return await res.arrayBuffer();
      }
    } catch (e) {
      if (hit) {
        try {
          return await hit.arrayBuffer();
        } catch {
          /* fall through to throw original */
        }
      }
      throw e;
    }
    if (hit) {
      try {
        return await hit.arrayBuffer();
      } catch {
        /* fall through to direct error below */
      }
    }
    throw new Error(`fetch failed for ${url}`);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch failed for ${url}: ${res.status}`);
  return await res.arrayBuffer();
}

async function fetchJson(url: string): Promise<unknown> {
  const buf = await fetchArrayBuffer(url);
  return JSON.parse(new TextDecoder().decode(buf));
}

export interface NodeBundle {
  dir: string;
  cfg: any;
  tokenizerJson: unknown | null;
}

export async function loadNodeBundle(
  modelDirOrRepo: string,
  opts?: { subfolder?: string | null; localDir?: string; token?: string | null },
): Promise<NodeBundle> {
  const fs: typeof import("node:fs/promises") = await import("node:fs/promises");
  const path: typeof import("node:path") = await import("node:path");
  const os: typeof import("node:os") = await import("node:os");
  const sub = opts?.subfolder ?? null;
  let dir = opts?.localDir ?? modelDirOrRepo;
  try {
    const st = await fs.stat(sub ? path.join(dir, sub) : dir);
    if (st.isDirectory()) dir = sub ? path.join(dir, sub) : dir;
    else dir = path.dirname(dir);
  } catch {
    const cache = path.join(
      os.homedir(),
      ".cache",
      "laya-ts",
      "hf",
      modelDirOrRepo.replace(/\//g, "__"),
      sub ?? "root",
    );
    await fs.mkdir(cache, { recursive: true });
    const token =
      opts?.token ?? (typeof process !== "undefined" ? (process as any).env?.["HF_TOKEN"] : undefined);
    for (const f of ["rl_agent_config.json", "tokenizer.json", "tokenizer/tokenizer.json", "encoder.onnx", "head.onnx"]) {
      try {
        await fs.stat(path.join(cache, f));
      } catch {
        const url = `https://huggingface.co/${modelDirOrRepo}/resolve/main/${sub ? sub + "/" : ""}${f}`;
        const res = await fetch(url, token ? { headers: { Authorization: `Bearer ${token}` } } : undefined);
        if (!res.ok) {
          if (f === "rl_agent_config.json") {
            throw new Error(
              `Incompatible model: ${JSON.stringify(modelDirOrRepo)} does not contain 'rl_agent_config.json'.`,
            );
          }
          continue;
        }
        const target = path.join(cache, f);
        await fs.mkdir(path.dirname(target), { recursive: true });
        await fs.writeFile(target, new Uint8Array(await res.arrayBuffer()));
      }
    }
    dir = cache;
  }
  let cfg: any = {};
  try {
    cfg = JSON.parse(await fs.readFile(path.join(dir, "rl_agent_config.json"), "utf8"));
  } catch {
    throw new Error(
      `Incompatible model: ${JSON.stringify(modelDirOrRepo)} does not contain 'rl_agent_config.json'.`,
    );
  }
  let tokenizerJson: unknown | null = null;
  for (const candidate of ["tokenizer.json", "tokenizer/tokenizer.json"]) {
    try {
      tokenizerJson = JSON.parse(await fs.readFile(path.join(dir, candidate), "utf8"));
      break;
    } catch {
      // Try the next supported Hugging Face layout.
    }
  }
  return { dir, cfg, tokenizerJson };
}

export interface WebBundle {
  dir: string;
  cfg: any;
  tokenizerJson: unknown | null;
}

function baseUrlFor(repoOrUrl: string, subfolder?: string | null): string {
  const sub = subfolder ? `/${subfolder.replace(/^\/+|\/+$/g, "")}` : "";
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(repoOrUrl)) {
    return `${repoOrUrl.replace(/\/+$/, "")}${sub}`;
  }
  return `https://huggingface.co/${repoOrUrl}/resolve/main${sub}`;
}

export async function loadWebBundle(
  repoOrUrl: string,
  opts?: { subfolder?: string | null },
): Promise<WebBundle> {
  const base = baseUrlFor(repoOrUrl, opts?.subfolder ?? null);
  let cfg: any;
  try {
    cfg = await fetchJson(`${base}/rl_agent_config.json`);
  } catch {
    throw new Error(`Incompatible model: ${JSON.stringify(repoOrUrl)} does not contain 'rl_agent_config.json'.`);
  }
  let tokenizerJson: unknown | null = null;
  for (const candidate of ["tokenizer.json", "tokenizer/tokenizer.json"]) {
    try {
      tokenizerJson = await fetchJson(`${base}/${candidate}`);
      break;
    } catch {
      // Try the next supported Hugging Face layout.
    }
  }
  return { dir: base, cfg, tokenizerJson };
}

export async function createNodeProvider(
  modelDir: string,
  opts?: ProviderOptions,
): Promise<SessionProvider> {
  const spec = "onnxruntime-" + "node";
  const ort: any = await import(/* @vite-ignore */ spec);
  applyNumThreads(ort, opts?.numThreads);
  try {
    const fs: typeof import("node:fs/promises") = await import("node:fs/promises");
    const path: typeof import("node:path") = await import("node:path");
    for (const f of ["encoder.onnx", "head.onnx"]) {
      const p = path.join(modelDir, f);
      try {
        await fs.stat(p);
      } catch {
        throw new Error(`Incompatible model: '${f}' not found in ${JSON.stringify(modelDir)} (expected ${p}).`);
      }
    }
  } catch (e) {
    if (e instanceof Error && e.message.includes("not found in")) throw e;
  }
  const dev = String(opts?.device ?? "cpu").toLowerCase();
  const want = dev === "cuda" ? "cuda" : dev === "dml" ? "dml" : "cpu";
  const make = async (ep: string) => {
    const e = await ort.InferenceSession.create(`${modelDir}/encoder.onnx`, {
      executionProviders: [ep],
    });
    const h = await ort.InferenceSession.create(`${modelDir}/head.onnx`, {
      executionProviders: ["cpu"],
    });
    return { e, h };
  };
  let enc: any;
  let head: any;
  let activeEP = want;
  try {
    ({ e: enc, h: head } = await make(want));
  } catch (e) {
    if (want !== "cpu") {
      console.warn(`Warning: ${want.toUpperCase()} requested but not available. Falling back to CPU.`);
      ({ e: enc, h: head } = await make("cpu"));
      activeEP = "cpu";
    } else {
      throw e;
    }
  }
  let cpuEnc: any = null;
  let cpuHead: any = null;
  const ensureCpu = async () => {
    if (!cpuEnc) {
      cpuEnc = await ort.InferenceSession.create(`${modelDir}/encoder.onnx`, {
        executionProviders: ["cpu"],
      });
      cpuHead = await ort.InferenceSession.create(`${modelDir}/head.onnx`, {
        executionProviders: ["cpu"],
      });
    }
    return { cpuEnc, cpuHead };
  };
  const runWithCpuFallback = async <T>(fn: (e: any, h: any) => Promise<T>): Promise<T> => {
    try {
      return await fn(enc, head);
    } catch (e) {
      if (activeEP !== "cpu" && isOomError(e)) {
        console.warn("Warning: GPU memory exceeded during inference. Falling back to CPU...");
        const { cpuEnc: ce, cpuHead: ch } = await ensureCpu();
        enc = ce;
        head = ch;
        activeEP = "cpu";
        return await fn(enc, head);
      }
      if (isOomError(e)) {
        throw new Error(`${(e as Error).message} (GPU out of memory; try device: "cpu")`);
      }
      throw e;
    }
  };
  return {
    runEncoder: async (b) =>
      runWithCpuFallback(async (e) => {
        const out = await e.run(feed(ort, b));
        const t = pickOutput(out, ["last_hidden_state", "lastHidden", "hidden_states"]);
        return { lastHidden: toNested(t.data, t.dims) };
      }),
    runHead: async (h, b) =>
      runWithCpuFallback(async (_e, hd) => {
        const out = await hd.run(feedHead(ort, h, b));
        const vals = Object.values(out) as any[];
        const lt = pickOutput(out, ["logits"]);
        const at = pickOutput(out, ["act_logits", "act"]) ?? vals[1] ?? vals[0];
        return { logits: toNested(lt.data, lt.dims), act: toNested(at.data, at.dims) };
      }),
  };
}

export async function createWebProvider(
  modelUrl: string,
  opts?: ProviderOptions,
): Promise<SessionProvider> {
  const spec = "onnxruntime-" + "web";
  const ort: any = await import(/* @vite-ignore */ spec);
  applyNumThreads(ort, opts?.numThreads);
  const base = modelUrl.replace(/\/+$/, "");
  const encUrl = `${base}/encoder.onnx`;
  const headUrl = `${base}/head.onnx`;
  let encBuf: ArrayBuffer;
  try {
    encBuf = await fetchArrayBuffer(encUrl);
  } catch {
    throw new Error(`Incompatible model: 'encoder.onnx' not found (expected ${encUrl}).`);
  }
  let headBuf: ArrayBuffer;
  try {
    headBuf = await fetchArrayBuffer(headUrl);
  } catch {
    throw new Error(`Incompatible model: 'head.onnx' not found (expected ${headUrl}).`);
  }
  let enc: any;
  try {
    enc = await ort.InferenceSession.create(new Uint8Array(encBuf), {
      executionProviders: ["webgpu", "wasm"],
    });
  } catch (e) {
    enc = await ort.InferenceSession.create(new Uint8Array(encBuf), {
      executionProviders: ["wasm"],
    });
  }
  const head = await ort.InferenceSession.create(new Uint8Array(headBuf), {
    executionProviders: ["wasm"],
  });
  return {
    runEncoder: async (b) => {
      try {
        const out = await enc.run(feed(ort, b));
        const t = pickOutput(out, ["last_hidden_state", "lastHidden", "hidden_states"]);
        return { lastHidden: toNested(t.data, t.dims) };
      } catch (e) {
        if (isOomError(e)) throw new Error(`${(e as Error).message} (WebGPU out of memory; WASM fallback already active)`);
        throw e;
      }
    },
    runHead: async (h, b) => {
      try {
        const out = await head.run(feedHead(ort, h, b));
        const vals = Object.values(out) as any[];
        const lt = pickOutput(out, ["logits"]);
        const at = pickOutput(out, ["act_logits", "act"]) ?? vals[1] ?? vals[0];
        return { logits: toNested(lt.data, lt.dims), act: toNested(at.data, at.dims) };
      } catch (e) {
        if (isOomError(e)) throw new Error(`${(e as Error).message} (out of memory; try fewer questions per call)`);
        throw e;
      }
    },
  };
}
