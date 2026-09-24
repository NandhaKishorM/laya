import { describe, expect, it, vi, afterEach } from "vitest";
import { loadWebBundle } from "../src/providers.js";

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
