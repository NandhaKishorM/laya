import { describe, expect, it, vi, afterEach } from "vitest";

const ortMocks = vi.hoisted(() => ({ create: vi.fn() }));

vi.mock("onnxruntime-web", () => ({
  Tensor: class {
    constructor(
      public type: string,
      public data: unknown,
      public dims: number[],
    ) {}
  },
  env: {},
  InferenceSession: { create: (...args: unknown[]) => ortMocks.create(...args) },
}));

import { createWebProvider } from "../src/providers.js";

const batch: any = {
  inputIds: [[1]],
  attentionMask: [[1]],
  markerPos: [[0]],
  markerMask: [[true]],
  qtype: [0],
};

function fetchStub() {
  const calls: string[] = [];
  const files: Record<string, Uint8Array> = {
    "https://example.com/m/encoder.onnx": new Uint8Array([1, 2, 3]),
    "https://example.com/m/encoder.onnx.data": new Uint8Array([4, 5]),
    "https://example.com/m/head.onnx": new Uint8Array([6]),
  };
  const fetchMock = vi.fn(async (url: string) => {
    calls.push(url);
    const body = files[url];
    if (!body) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
    return { ok: true, status: 200, arrayBuffer: async () => body.buffer.slice(0) as ArrayBuffer };
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("web provider external data", () => {
  it("mounts the sidecar via externalData", async () => {
    fetchStub();
    ortMocks.create.mockImplementation(async (_buf: unknown, opts: any) => ({
      run: async () => ({ lastHidden: { data: new Float32Array([1, 2]), dims: [1, 1, 2] } }),
      usedOpts: opts,
    }));
    await createWebProvider("https://example.com/m");
    const encCall = ortMocks.create.mock.calls.find((c) =>
      (c[1] as any)?.executionProviders?.includes("webgpu"),
    );
    expect(encCall).toBeDefined();
    const ext = (encCall![1] as any).externalData;
    expect(ext).toHaveLength(1);
    expect(ext[0].path).toBe("encoder.onnx.data");
    expect(Array.from(ext[0].data as Uint8Array)).toEqual([4, 5]);
  });

  it("re-reads encoder bytes on a [WebGPU] run failure and sticks with WASM", async () => {
    const { calls } = fetchStub();
    const webgpuErr = new Error('[WebGPU] Kernel "SkipLayerNormalization" failed. Error: Beta must be 1D');
    const goodRun = async () => ({ lastHidden: { data: new Float32Array([1, 2]), dims: [1, 1, 2] } });
    ortMocks.create.mockImplementation(async (_buf: unknown, opts: any) =>
      (opts as any)?.executionProviders?.includes("webgpu")
        ? { run: async () => { throw webgpuErr; } }
        : { run: goodRun },
    );
    const provider = await createWebProvider("https://example.com/m");
    const encFetches = () => calls.filter((u) => u === "https://example.com/m/encoder.onnx").length;
    expect(encFetches()).toBe(1);
    const out = await provider.runEncoder(batch);
    expect(out.lastHidden).toEqual([[[1, 2]]]);
    // Fallback re-reads instead of reusing 1GB+ retained buffers.
    expect(encFetches()).toBe(2);
    const wasmCreates = ortMocks.create.mock.calls.filter(
      (c) => !(c[1] as any)?.executionProviders?.includes("webgpu"),
    );
    expect(wasmCreates.length).toBeGreaterThanOrEqual(2); // head + fallback encoder
    const encWasm = wasmCreates.find((c) => (c[1] as any)?.externalData?.[0]?.path === "encoder.onnx.data");
    expect(encWasm).toBeDefined();
    // Second run goes straight to WASM: no new session, no new fetch.
    await provider.runEncoder(batch);
    expect(encFetches()).toBe(2);
  });
});
