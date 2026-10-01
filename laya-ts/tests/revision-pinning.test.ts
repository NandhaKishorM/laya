import { afterEach, describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Agent } from "../src/agent.js";
import { Router } from "../src/router.js";
import { PINNED_REVISIONS, createNodeProvider, loadNodeBundle, loadWebBundle, resolveRevision } from "../src/providers.js";

const fakeProvider = () => ({
  async runEncoder(_b: any) { return { lastHidden: [[[1, 0], [0, 1]]] }; },
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

describe("Router digest tests", () => {
  it("Router normalises digest keys like revision keys", () => {
    const router = new Router({
      sha256Digests: { ml: { "w.bin": "a".repeat(64) }, typed: null },
    });
    expect(Object.keys(router.sha256Digests).sort()).toEqual(["multilingual", "typed-decisions"]);
    expect(router.sha256_digests).toBe(router.sha256Digests);
    expect(new Router().sha256Digests).toEqual({});
  });

  it("snake_case sha256_digests alias is accepted", () => {
    const router = new Router({
      sha256_digests: { en: { "w.bin": "a".repeat(64) } },
    });
    expect(router.sha256Digests.english).toEqual({ "w.bin": "a".repeat(64) });
  });

  it("misspelled model fails at construction", () => {
    expect(() => new Router({ sha256Digests: { engligh: { "w.bin": "a".repeat(64) } } }))
      .toThrow(/unknown model "engligh"/);
  });

  it("each checkpoint gets its own expectedSha256 map upon load", async () => {
    const captured: Array<{ repo: string; opts: Record<string, unknown> | undefined }> = [];
    const spy = vi.spyOn(Agent, "load").mockImplementation(async (repo, opts) => {
      captured.push({ repo, opts });
      return new Agent({ provider: fakeProvider(), cfg: {} });
    });
    try {
      const router = new Router({
        sha256Digests: {
          english: { "model.safetensors": "a".repeat(64) },
          multilingual: { "model.safetensors": "b".repeat(64) },
        },
      });
      await router.load("english");
      await router.load("multilingual");
      expect(captured[0].opts?.expectedSha256).toEqual({ "model.safetensors": "a".repeat(64) });
      expect(captured[1].opts?.expectedSha256).toEqual({ "model.safetensors": "b".repeat(64) });
    } finally {
      spy.mockRestore();
    }
  });

  it("unlisted checkpoint is left without expectedSha256", async () => {
    const captured: Array<{ repo: string; opts: Record<string, unknown> | undefined }> = [];
    const spy = vi.spyOn(Agent, "load").mockImplementation(async (repo, opts) => {
      captured.push({ repo, opts });
      return new Agent({ provider: fakeProvider(), cfg: {} });
    });
    try {
      const router = new Router({
        sha256Digests: { multilingual: { "model.safetensors": "b".repeat(64) } },
      });
      await router.load("english");
      expect(captured[0].opts?.expectedSha256).toBeUndefined();
    } finally {
      spy.mockRestore();
    }
  });

  it("none/null entry passes an empty map to mask environment defaults", async () => {
    const captured: Array<{ repo: string; opts: Record<string, unknown> | undefined }> = [];
    const spy = vi.spyOn(Agent, "load").mockImplementation(async (repo, opts) => {
      captured.push({ repo, opts });
      return new Agent({ provider: fakeProvider(), cfg: {} });
    });
    try {
      const router = new Router({
        sha256Digests: { english: null },
      });
      await router.load("english");
      expect(captured[0].opts?.expectedSha256).toEqual({});
    } finally {
      spy.mockRestore();
    }
  });
});

describe("Router env digest tests", () => {
  const origEnv = process.env["LAYA_SHA256_DIGESTS"];

  afterEach(() => {
    if (origEnv === undefined) {
      delete process.env["LAYA_SHA256_DIGESTS"];
    } else {
      process.env["LAYA_SHA256_DIGESTS"] = origEnv;
    }
  });

  it("flat map is left to underlying provider", () => {
    process.env["LAYA_SHA256_DIGESTS"] = JSON.stringify({ "model.safetensors": "a".repeat(64) });
    const router = new Router();
    expect(router.sha256Digests).toEqual({});
  });

  it("nested map is split per checkpoint and unlisted models default to {}", () => {
    const engDigests = { "model.safetensors": "a".repeat(64) };
    const multiDigests = { "model.safetensors": "b".repeat(64) };
    process.env["LAYA_SHA256_DIGESTS"] = JSON.stringify({
      english: engDigests,
      multilingual: multiDigests,
    });
    const router = new Router();
    expect(Object.keys(router.sha256Digests).sort()).toEqual(["english", "multilingual", "typed-decisions"]);
    expect(router.sha256Digests.english).toEqual(engDigests);
    expect(router.sha256Digests.multilingual).toEqual(multiDigests);
    expect(router.sha256Digests["typed-decisions"]).toEqual({});
  });

  it("nested map keys are normalised", () => {
    const engDigests = { "model.safetensors": "a".repeat(64) };
    process.env["LAYA_SHA256_DIGESTS"] = JSON.stringify({ en: engDigests });
    const router = new Router();
    expect(router.sha256Digests.english).toEqual(engDigests);
  });

  it("constructor argument wins for the checkpoint it names", () => {
    const engDigests = { "model.safetensors": "a".repeat(64) };
    const multiDigests = { "model.safetensors": "b".repeat(64) };
    process.env["LAYA_SHA256_DIGESTS"] = JSON.stringify({
      english: engDigests,
      multilingual: multiDigests,
    });
    const router = new Router({
      sha256Digests: { english: { "model.safetensors": "c".repeat(64) } },
    });
    expect(router.sha256Digests.english).toEqual({ "model.safetensors": "c".repeat(64) });
    expect(router.sha256Digests.multilingual).toEqual(multiDigests);
  });

  it("misspelled key in environment fails at construction", () => {
    process.env["LAYA_SHA256_DIGESTS"] = JSON.stringify({ engligh: { "w.bin": "a".repeat(64) } });
    expect(() => new Router()).toThrow(/unknown model "engligh"/);
  });

  it("mixed artifact and model keys fail rather than guess", () => {
    process.env["LAYA_SHA256_DIGESTS"] = JSON.stringify({
      "model.safetensors": "a".repeat(64),
      english: { "model.safetensors": "a".repeat(64) },
    });
    expect(() => new Router()).toThrow(/LAYA_SHA256_DIGESTS/);
  });

  it("unparseable environment returns empty without throwing", () => {
    for (const val of ["", "   ", "{not json", "[]", '"digests"', "{}"]) {
      process.env["LAYA_SHA256_DIGESTS"] = val;
      expect(new Router().sha256Digests).toEqual({});
    }
  });
});
