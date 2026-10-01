import { describe, expect, it, vi, afterEach } from "vitest";
import { createWebProvider, loadWebBundle } from "../src/providers.js";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(obj: unknown, ok = true, status = 200) {
  const text = JSON.stringify(obj);
  const buf = new TextEncoder().encode(text).buffer as ArrayBuffer;
  return {
    ok,
    status,
    clone: () => jsonResponse(obj, ok, status),
    arrayBuffer: async () => buf.slice(0),
  };
}

describe("load robustness", () => {
  it("retries transient 503 then succeeds", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 503, arrayBuffer: async () => new ArrayBuffer(0), clone: () => ({}) })
      .mockResolvedValueOnce(jsonResponse({ max_len: 512 }))
      .mockResolvedValueOnce(jsonResponse({ model: { vocab: {}, merges: [] }, added_tokens: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const progress: string[] = [];
    const bundle = await loadWebBundle("https://example.com/m", {
      onProgress: (_d: number, _t: number, f: string) => progress.push(f),
    } as any);
    expect(bundle.cfg).toEqual({ max_len: 512 });
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(3);
    expect(progress.length).toBeGreaterThan(0);
  });
  it("aborted signal throws without retry", async () => {
    const c = new AbortController();
    c.abort();
    vi.stubGlobal("fetch", vi.fn(async (_u: any, init: any) => {
      if (init?.signal?.aborted ?? c.signal.aborted) throw new DOMException("aborted", "AbortError");
      return jsonResponse({});
    }));
    await expect(loadWebBundle("https://example.com/m", { signal: c.signal } as any)).rejects.toThrow();
  });
});

describe("web provider downloads", () => {
  function onnxBytes() {
    return {
      ok: true,
      status: 200,
      clone: () => ({}),
      arrayBuffer: async () => new Uint8Array([0, 1, 2, 3]).buffer as ArrayBuffer,
    };
  }
  it("passes signal to the model fetches and reports per-file progress", async () => {
    const seen: Array<{ url: string; signal: unknown }> = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: any) => {
      seen.push({ url, signal: init?.signal ?? null });
      return onnxBytes();
    }));
    const progress: Array<[number, number, string]> = [];
    const c = new AbortController();
    // Garbage bytes: the real runtime rejects at session creation, after the
    // fetches already carried the signal and the progress fired.
    await expect(
      createWebProvider("https://example.com/m", {
        signal: c.signal,
        onProgress: (d: number, t: number, f: string) => progress.push([d, t, f]),
      } as any),
    ).rejects.toThrow();
    const encFetch = seen.find((s) => s.url.endsWith("encoder.onnx"));
    const headFetch = seen.find((s) => s.url.endsWith("head.onnx"));
    expect(encFetch?.signal).toBe(c.signal);
    expect(headFetch?.signal).toBe(c.signal);
    expect(progress).toEqual([
      [1, 2, "encoder.onnx"],
      [2, 2, "head.onnx"],
    ]);
  });
});

describe("load abort semantics", () => {
  it("aborted signal surfaces AbortError, not Incompatible model", async () => {
    const c = new AbortController();
    c.abort();
    vi.stubGlobal("fetch", vi.fn(async (_u: any, init: any) => {
      if (init?.signal?.aborted ?? c.signal.aborted) throw new DOMException("aborted", "AbortError");
      return jsonResponse({});
    }));
    const err = await loadWebBundle("https://example.com/m", { signal: c.signal } as any).catch((e) => e);
    expect(err?.name).toBe("AbortError");
  });
});
