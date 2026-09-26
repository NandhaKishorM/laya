import { afterEach, describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Agent } from "../src/agent.js";
import { Router } from "../src/router.js";
import { PINNED_REVISIONS, createNodeProvider, loadNodeBundle, loadWebBundle, resolveRevision } from "../src/providers.js";

const fakeProvider = () => ({
  async runEncoder(_b: any) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: any) { return { logits: [[2, 0]], act: [[3, 0]] }; },
});

class StubResponse {
  constructor(public body: Uint8Array | null, public status = 200) {}
  get ok() { return this.status >= 200 && this.status < 300; }
  headers = { get: (n: string) => (n === "x-repo-commit" ? "abc123" : null) };
  async arrayBuffer() { return (this.body ?? new Uint8Array()).buffer as ArrayBuffer; }
}

const tmpDirs: string[] = [];
function makeCheckpoint(files: Record<string, Uint8Array>): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "laya-rev-"));
  tmpDirs.push(dir);
  for (const [rel, data] of Object.entries(files)) {
    const p = path.join(dir, rel);
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, data);
  }
  return dir;
}
const cfgFile = { "rl_agent_config.json": new TextEncoder().encode('{"act_costs":{"a":0}}') };
const sha256 = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");

afterEach(() => {
  vi.unstubAllGlobals();
  for (const d of tmpDirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
});

describe("resolveRevision", () => {
  it("explicit revision is returned", () => {
    expect(resolveRevision("convaiinnovations/laya", "abc123")).toBe("abc123");
  });
  it("published repos keep the hub default without an explicit pin", () => {
    for (const repo of Object.keys(PINNED_REVISIONS)) {
      expect(resolveRevision(repo)).toBeNull();
    }
  });
  it("unknown repos keep the hub default", () => {
    expect(resolveRevision("acme/custom-model")).toBeNull();
  });
});

describe("loadNodeBundle pinning", () => {
  it("keeps the hub default unless a revision is explicitly requested", async () => {
    const urls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      urls.push(url);
      return url.endsWith("rl_agent_config.json")
        ? new StubResponse(cfgFile["rl_agent_config.json"])
        : new StubResponse(null, 404);
    }));
    const cacheDir = path.join(os.homedir(), ".cache", "laya-ts", "hf", "convaiinnovations__laya", "root");
    try {
      const bundle = await loadNodeBundle("convaiinnovations/laya");
      expect(urls[0]).toContain("/resolve/main/");
      expect(bundle.dir).toBe(cacheDir);
      // The hub reports the exact commit served even when no pin was requested.
      expect(bundle.revision).toBe("abc123");
      expect(bundle.cfg.act_costs).toEqual({ a: 0 });
    } finally {
      fs.rmSync(path.join(os.homedir(), ".cache", "laya-ts", "hf", "convaiinnovations__laya"), { recursive: true, force: true });
    }
  });

  it("uses an explicit revision in both the URL and cache key", async () => {
    const urls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      urls.push(url);
      return url.endsWith("rl_agent_config.json")
        ? new StubResponse(cfgFile["rl_agent_config.json"])
        : new StubResponse(null, 404);
    }));
    try {
      const bundle = await loadNodeBundle("convaiinnovations/laya", { revision: "abc123" });
      expect(urls[0]).toContain("/resolve/abc123/");
      expect(bundle.dir).toContain(path.join("root", "abc123"));
    } finally {
      fs.rmSync(path.join(os.homedir(), ".cache", "laya-ts", "hf", "convaiinnovations__laya"), { recursive: true, force: true });
    }
  });

  it("loadWebBundle reports the x-repo-commit header", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      return url.endsWith("rl_agent_config.json")
        ? new StubResponse(cfgFile["rl_agent_config.json"])
        : new StubResponse(null, 404);
    }));
    const bundle = await loadWebBundle("convaiinnovations/laya");
    expect(bundle.revision).toBe("abc123");
  });

  it("unpinned repos keep the mutable main default", async () => {
    const urls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      urls.push(url);
      return url.endsWith("rl_agent_config.json")
        ? new StubResponse(cfgFile["rl_agent_config.json"])
        : new StubResponse(null, 404);
    }));
    try {
      await loadNodeBundle("acme/custom-model");
      expect(urls[0]).toContain("/resolve/main/");
    } finally {
      fs.rmSync(path.join(os.homedir(), ".cache", "laya-ts", "hf", "acme__custom-model"), { recursive: true, force: true });
    }
  });
});

describe("loadNodeBundle expectedSha256", () => {
  it("local dirs never download and report no revision", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const dir = makeCheckpoint(cfgFile);
    const bundle = await loadNodeBundle(dir);
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(bundle.revision).toBeNull();
  });

  it("matching digests pass; unlisted artifacts are unchecked", async () => {
    const weights = new TextEncoder().encode("weights");
    const dir = makeCheckpoint({ ...cfgFile, "model.bin": weights });
    const bundle = await loadNodeBundle(dir, {
      expectedSha256: { "model.bin": sha256(weights) },
    });
    expect(bundle.dir).toBe(dir);
  });

  it("mismatches refuse to load", async () => {
    const dir = makeCheckpoint({ ...cfgFile, "model.bin": new TextEncoder().encode("weights") });
    await expect(loadNodeBundle(dir, { expectedSha256: { "model.bin": "0".repeat(64) } }))
      .rejects.toThrow(/SHA-256 mismatch/);
  });

  it("missing artifacts refuse to load", async () => {
    const dir = makeCheckpoint(cfgFile);
    await expect(loadNodeBundle(dir, { expectedSha256: { "absent.bin": "0".repeat(64) } }))
      .rejects.toThrow(/cannot verify/);
  });

  it("escaping digest paths are rejected", async () => {
    const dir = makeCheckpoint(cfgFile);
    for (const rel of ["../evil", "..", "a/../../evil", "/absolute/evil", "C:\\absolute\\evil"]) {
      await expect(loadNodeBundle(dir, { expectedSha256: { [rel]: "0".repeat(64) } }))
        .rejects.toThrow(/unsafe (?:absolute )?path/);
    }
  });

  it("createNodeProvider refuses mismatched ONNX digests before creating sessions", async () => {
    const encoder = new TextEncoder().encode("encoder");
    const dir = makeCheckpoint({ "encoder.onnx": encoder, "head.onnx": new TextEncoder().encode("head") });
    await expect(createNodeProvider(dir, {
      expectedSha256: { "encoder.onnx": "0".repeat(64) },
    })).rejects.toThrow(/SHA-256 mismatch/);
  });
});

describe("revision plumbing", () => {
  it("Agent keeps the revision from its options", () => {
    const agent = new Agent({ provider: fakeProvider(), cfg: {}, revision: "abc123" });
    expect(agent.revision).toBe("abc123");
    expect(new Agent({ provider: fakeProvider(), cfg: {} }).revision).toBeNull();
  });

  it("Router stores global and per-model revisions", () => {
    const router = new Router({
      revision: "default",
      revisions: { ml: "multi-sha", typed: "typed-sha" },
    });
    expect(router.revision).toBe("default");
    expect(router.revisions.multilingual).toBe("multi-sha");
    expect(router.revisions["typed-decisions"]).toBe("typed-sha");
    expect(new Router().revision).toBeNull();
    expect(new Router().revisions).toEqual({});
  });
});
