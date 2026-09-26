import { describe, expect, it, vi } from "vitest";
import { Router, normaliseName } from "../src/router.js";
describe("router", () => {
  it("normalises aliases", () => { expect(normaliseName("en")).toBe("english"); });
  it("rejects unknown model", () => { expect(() => normaliseName("jev-1")).toThrow(); });
  it("routes hindi script to multilingual without loading", () => {
    const r = new Router();
    expect(r.route({ body: "मुझसे दो बार शुल्क लिया गया" }, {}).model).toBe("multilingual");
    expect(r.loaded).toEqual([]);
  });
  it("explicit model wins", () => {
    expect(new Router().route("hi", {}, { model: "typed-decisions" }).model).toBe("typed-decisions");
  });
  it("lang_guess callable routes", () => {
    const r = new Router();
    expect(r.route("text", {}, { langGuess: () => "ro" }).model).toBe("multilingual");
  });
  it("shares concurrent loads and emits lifecycle hooks once per cached model", async () => {
    const events: string[] = [];
    const loader = vi.fn(async (name: string) => ({ name }));
    const r = new Router({
      maxLoaded: 1,
      loader,
      hooks: [{
        onLoad: (ctx) => events.push(`load:${ctx.model}`),
        onEvict: (ctx) => events.push(`evict:${ctx.model}`),
      }],
    });

    const [first, second] = await Promise.all([r.load("en"), r.load("english")]);
    expect(first).toBe(second);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(r.loaded).toEqual(["english"]);
    expect(r._agents.size).toBe(1);
    await r.load("multilingual");
    expect(events).toEqual(["load:english", "evict:english", "load:multilingual"]);
    expect(r.loaded).toEqual(["multilingual"]);
  });
  it("clears an unsuccessful in-flight load so the model can be retried", async () => {
    const agent = { name: "english" };
    const loader = vi.fn()
      .mockRejectedValueOnce(new Error("load failed"))
      .mockResolvedValue(agent);
    const r = new Router({ loader });

    const results = await Promise.allSettled([r.load("en"), r.load("english")]);
    expect(results.map((result) => result.status)).toEqual(["rejected", "rejected"]);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(await r.load("english")).toBe(agent);
    expect(loader).toHaveBeenCalledTimes(2);
    expect(r.loaded).toEqual(["english"]);
  });
});
