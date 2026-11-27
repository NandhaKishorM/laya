import { describe, expect, it } from "vitest";
import { cached, cachedEmbedFn, shortlistChoice } from "../src/shortlist.js";

const emb = async (texts: string[]) =>
  texts.map((t) => (t.includes("refund") || t.includes("billed") ? [1, 0] : [0, 1]));

describe("shortlist", () => {
  it("keeps top-k by cosine", async () => {
    const labels = await shortlistChoice({ text: "billed twice refund" },
      { card_arrival: "where is my card", transfer_fee: "fee on transfer", refund: "refund money" },
      emb as any, 1);
    expect(labels).toEqual(["refund"]);
  });

  it("passthrough when k >= n skips embed", async () => {
    let called = false;
    await shortlistChoice("x", { a: "1" }, (async (t: string[]) => { called = true; return t.map(() => [1]); }) as any, 5);
    expect(called).toBe(false);
  });

  it("serializes structured instructions like Python", async () => {
    const calls: string[][] = [];
    const recordingEmbed = async (texts: string[]) => {
      calls.push([...texts]);
      return texts.map(() => [1]);
    };

    await shortlistChoice("<state>", { a: "one", b: "two" }, recordingEmbed, 1,
      { topic: "billing", tags: ["a", 1.5, true, null, "é"], max: 1e16, min: 0.00005 });
    expect(calls[0][0]).toBe(
      '{"topic": "billing", "tags": ["a", 1.5, true, null, "é"], "max": 1e+16, "min": 5e-05}\n<state>',
    );
  });
});

describe("cached embedder", () => {
  it("refuses non-callable embed_fn with TypeError", () => {
    expect(() => cached(null as any)).toThrow(TypeError);
    expect(() => cached(undefined as any)).toThrow(TypeError);
    expect(() => cached("not a function" as any)).toThrow(TypeError);
  });

  it("refuses non-positive or non-integer maxsize", () => {
    expect(() => cached(emb, 0)).toThrow(/maxsize must be a positive integer/);
    expect(() => cached(emb, -2)).toThrow(/maxsize must be a positive integer/);
    expect(() => cached(emb, 2.5)).toThrow(/maxsize must be a positive integer/);
    expect(() => cached(emb, "10" as any)).toThrow(/maxsize must be a positive integer/);
    expect(() => cached(emb, true as any)).toThrow(/maxsize must be a positive integer/);
  });

  it("records hits and misses and deduplicates cold lookups", async () => {
    const calls: string[][] = [];
    const countingEmbed = async (texts: string[]) => {
      calls.push([...texts]);
      return texts.map((t) => [t.length, 1]);
    };

    const c = cached(countingEmbed);
    expect(c.cacheInfo()).toEqual({ size: 0, maxsize: 4096, hits: 0, misses: 0 });

    // Cold call with duplicates
    const res1 = await c(["apple", "banana", "apple"]);
    expect(res1).toHaveLength(3);
    expect(res1[0]).toEqual([5, 1]);
    expect(res1[1]).toEqual([6, 1]);
    expect(res1[2]).toEqual([5, 1]);
    expect(calls).toEqual([["apple", "banana"]]); // deduplicated miss query
    expect(c.cacheInfo()).toEqual({ size: 2, maxsize: 4096, hits: 0, misses: 3 });

    // Second call: pure hits
    const res2 = await c(["banana", "apple"]);
    expect(res2).toHaveLength(2);
    expect(calls).toHaveLength(1); // no new embed calls
    expect(c.cacheInfo()).toEqual({ size: 2, maxsize: 4096, hits: 2, misses: 3 });

    // Mixed call: 1 hit, 1 miss
    const res3 = await c(["apple", "cherry"]);
    expect(res3).toHaveLength(2);
    expect(calls).toEqual([["apple", "banana"], ["cherry"]]);
    expect(c.cacheInfo()).toEqual({ size: 3, maxsize: 4096, hits: 3, misses: 4 });
  });

  it("evicts least-recently-used entries when size exceeds maxsize", async () => {
    const calls: string[][] = [];
    const tracking = async (texts: string[]) => {
      calls.push([...texts]);
      return texts.map((t) => [t.length]);
    };

    const c = cachedEmbedFn(tracking, 2);

    await c(["a", "b"]); // cache: [a, b]
    expect(c.cacheInfo().size).toBe(2);

    await c(["a"]); // access a -> LRU order: [b, a]
    expect(calls.length).toBe(1);
    expect(c.cacheInfo().hits).toBe(1);

    await c(["c"]); // adds c -> evicts b! cache: [a, c]
    expect(calls[calls.length - 1]).toEqual(["c"]);
    expect(c.cacheInfo().size).toBe(2);

    // both survivors are in cache: hits
    await c(["a", "c"]);
    expect(calls.length).toBe(2);

    // b was evicted, so it re-embeds
    await c(["b"]);
    expect(calls[calls.length - 1]).toEqual(["b"]);
    expect(c.cacheInfo().size).toBe(2);

    // most recent pair [c, b] survives
    await c(["c", "b"]);
    expect(calls.length).toBe(3);

    // a was now the oldest of the three and was evicted when b was inserted
    await c(["a"]);
    expect(calls[calls.length - 1]).toEqual(["a"]);
    expect(calls.length).toBe(4);
  });

  it("resets cache and counters on cacheClear() and cache_clear()", async () => {
    const c = cached(async (texts) => texts.map(() => [1, 2]));
    await c(["hello", "world"]);
    expect(c.cacheInfo().size).toBe(2);
    expect(c.cacheInfo().misses).toBe(2);

    c.cacheClear();
    expect(c.cacheInfo()).toEqual({ size: 0, maxsize: 4096, hits: 0, misses: 0 });
    expect(c.cache_info()).toEqual({ size: 0, maxsize: 4096, hits: 0, misses: 0 });

    c.cache_clear();
    expect(c.cacheInfo().size).toBe(0);
  });

  it("handles empty arrays and normalizes null/undefined keys", async () => {
    let called = false;
    const c = cached(async (t) => { called = true; return t.map(() => [1]); });

    const empty = await c([]);
    expect(empty).toEqual([]);
    expect(called).toBe(false);

    const normed = await c([null as any, undefined as any, ""]);
    expect(normed).toHaveLength(3);
    expect(c.cacheInfo().size).toBe(1); // all normalized to ""
    expect(c.cacheInfo().misses).toBe(3);
  });

  it("defensively copies cached vectors against caller mutations", async () => {
    const c = cached(async (texts) => texts.map(() => [10, 20]));
    const first = await c(["immutable"]);
    first[0][0] = 999;

    const second = await c(["immutable"]);
    expect(second[0][0]).toBe(10);
  });

  it("refuses changed embedding dimensions with descriptive error", async () => {
    let dim = 2;
    const changing = async (texts: string[]) => texts.map(() => Array(dim).fill(1));
    const c = cached(changing);

    await c(["first"]); // caches dim 2

    dim = 4;
    await expect(c(["second"])).rejects.toThrow(
      /embedder returned dim 4, but the cache holds dim 2; call cacheClear\(\)/,
    );
  });

  it("does not poison the cache on transient embedder failure", async () => {
    let fail = true;
    const flaky = async (texts: string[]) => {
      if (fail) throw new Error("network down");
      return texts.map(() => [1, 2]);
    };

    const c = cached(flaky);
    await expect(c(["retry_me"])).rejects.toThrow("network down");
    expect(c.cacheInfo().size).toBe(0);

    fail = false;
    const ok = await c(["retry_me"]);
    expect(ok).toEqual([[1, 2]]);
    expect(c.cacheInfo().size).toBe(1);
  });

  it("integrates seamlessly with shortlistChoice across repeated calls", async () => {
    let embedCalls = 0;
    const embedFn = async (texts: string[]) => {
      embedCalls++;
      return texts.map((t) => (t.includes("refund") ? [1, 0] : [0, 1]));
    };

    const cachedEmbed = cached(embedFn);
    const criteria = { card: "card arrival", refund: "refund money", fee: "transfer fee" };

    // First call: embeds query + 3 criteria (4 texts)
    const first = await shortlistChoice("first query refund", criteria, cachedEmbed, 1);
    expect(first).toEqual(["refund"]);
    const firstCalls = embedCalls;
    const firstHits = cachedEmbed.cacheInfo().hits;

    // Second call: criteria is cached! Only embeds the new query (1 text)
    const second = await shortlistChoice("second query refund", criteria, cachedEmbed, 1);
    expect(second).toEqual(["refund"]);
    expect(embedCalls).toBe(firstCalls + 1);
    expect(cachedEmbed.cacheInfo().hits - firstHits).toBe(3);
  });

  it("applies LRU recency in input order so missing keys before cached keys do not displace later inputs", async () => {
    let reembedded = false;
    const tracking = cachedEmbedFn(async (texts) => {
      reembedded = true;
      return texts.map((t) => [t.length]);
    }, 2);

    await tracking(["a", "b"]); // cache has [a, b]
    expect(tracking.cacheInfo().size).toBe(2);

    // "q" is missing, "a" and "b" are cached.
    // In input order: "q" is evaluated first (evicts a -> [b, q]),
    // then "a" is accessed (evicts b -> [q, a]),
    // then "b" is accessed (evicts q -> [a, b]).
    await tracking(["q", "a", "b"]);
    expect(tracking.cacheInfo().size).toBe(2);

    // Calling [a, b] should be complete hits without re-embedding
    reembedded = false;
    await tracking(["a", "b"]);
    expect(reembedded).toBe(false);
  });

  it("validates all fresh rows for dimension consistency before caching", async () => {
    // Case 1: Inconsistent dimensions from fresh call on empty cache
    const inconsistent = cached(async (texts) => texts.map((t, i) => Array(i + 1).fill(1)));
    await expect(inconsistent(["one", "two"])).rejects.toThrow(
      /inconsistent embedding dimensions/,
    );
    expect(inconsistent.cacheInfo().size).toBe(0);

    // Case 2: fresh[0] has matching dim but fresh[1] has differing dim on populated cache
    let secondCalls = 0;
    const mixed = cached(async (texts) => {
      secondCalls++;
      if (secondCalls === 1) return texts.map(() => [1, 2]);
      return [[1, 2], [1, 2, 3]]; // fresh[0] is dim 2, fresh[1] is dim 3
    });
    await mixed(["first"]);
    await expect(mixed(["good", "bad"])).rejects.toThrow(
      /embedder returned dim 3, but the cache holds dim 2; call cacheClear\(\)/,
    );
  });

  it("prevents in-flight embeddings from restoring entries into a cleared cache", async () => {
    let resolveEmbed: ((val: number[][]) => void) | null = null;
    const delayed = cached(
      () =>
        new Promise<number[][]>((res) => {
          resolveEmbed = res;
        }),
    );

    const pending = delayed(["slow"]);
    expect(delayed.cacheInfo().size).toBe(0);

    // Clear cache while request is in-flight
    delayed.cacheClear();

    // Resolve the delayed embeddings
    resolveEmbed!([[42]]);
    const res = await pending;
    expect(res).toEqual([[42]]);

    // Cache should NOT be populated with the stale in-flight result
    expect(delayed.cacheInfo().size).toBe(0);
  });
});
