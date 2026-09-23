import { describe, expect, it } from "vitest";
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
});
