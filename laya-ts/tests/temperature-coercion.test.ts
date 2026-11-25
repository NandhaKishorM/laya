import { describe, expect, it, vi } from "vitest";
import { clampTemperature, TEMP_MAX, TEMP_MIN } from "../src/common.js";
import { Agent } from "../src/agent.js";

// The reference asks "is this a number?" with `float()`, which refuses a string that holds nothing
// but blanks and refuses any list. `Number()` reads `Number(" ")` and `Number([])` as 0, which the
// clamp bounds to TEMP_MIN, and `Number([2])` as 2, which sits inside the range and so passes for a
// fitted temperature. 0 is the worse of the two: a temperature below 1 sharpens the logits, so the
// published confidence is inflated rather than merely uncalibrated. Those four are the divergences
// measured as `clamp_temperature(t)` / `clampTemperature(t)` on the port's own base; the rest of the
// table pins the fallbacks the port already got right, so a future coercion shortcut is caught here.

const fakeProvider = () => ({
  async runEncoder(_b: any) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: any) { return { logits: [[2, 0]], act: [[3, 0]] }; },
});

const CHOICE = { d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } } } as any;

function loadWith(temperature: unknown[]) {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  try {
    const agent = new Agent({
      provider: fakeProvider(),
      cfg: { temperature },
    } as any);
    return { agent, warnings: warn.mock.calls.map((call: unknown[]) => String(call[0])) };
  } finally {
    warn.mockRestore();
  }
}

describe("clampTemperature", () => {
  it.each([
    ["[]", []],
    ["[2]", [2]],
    ['["2"]', ["2"]],
    ['" "', " "],
    ['"\\t\\n"', "\t\n"],
    ["null", null],
    ["undefined", undefined],
    ['""', ""],
    ["true", true],
    ["false", false],
    ["{}", {}],
    ["NaN", NaN],
    ["Infinity", Infinity],
  ])("falls back to the neutral 1.0 for %s, which is not a number", (_label, value) => {
    expect(clampTemperature(value)).toBe(1.0);
  });

  it("still bounds a real number instead of falling back", () => {
    // The fallback must not swallow the clamp: 0 is a number, and 0 means sharpen hard.
    expect(clampTemperature(0)).toBe(TEMP_MIN);
    expect(clampTemperature(-3)).toBe(TEMP_MIN);
    expect(clampTemperature(0.1006)).toBe(TEMP_MIN);
    expect(clampTemperature(9)).toBe(TEMP_MAX);
    expect(clampTemperature(1.7601518630981445)).toBe(1.7601518630981445);
    expect(clampTemperature(1.0)).toBe(1.0);
  });

  it("still reads a number out of a string, as float() does", () => {
    expect(clampTemperature("2")).toBe(2.0);
    expect(clampTemperature(" 2 ")).toBe(2.0);
    expect(clampTemperature("2e0")).toBe(2.0);
    expect(clampTemperature("1.5")).toBe(1.5);
    // float("nan") and float("inf") do convert, and then fail the finite test the same way.
    expect(clampTemperature("nan")).toBe(1.0);
    expect(clampTemperature("invalid")).toBe(1.0);
    // A bigint is Python's int on the JSON-bigint path, not a value to fall back for.
    expect(clampTemperature(2n as never)).toBe(2.0);
  });
});

describe("checkpoint temperature loading", () => {
  it.each([
    [["[2]", [2]]],
    [['["2"]', ["2"]]],
    [['" "', " "]],
    [["[]", []]],
  ])("loads %s as the neutral 1.0 and says so", (_label: string, value: unknown) => {
    const { agent, warnings } = loadWith([value, 2.0, 3.0]);
    expect(agent.temperature).toEqual([1.0, 2.0, 3.0]);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain("temperature[0]");
    expect(warnings[0]).toContain("uncalibrated");
  });

  it("scales the answer by the neutral 1.0 it installed, not by a coerced number", async () => {
    // The whole point of the fallback. Logits [2, 0] at T=1 publish 0.8808; the same forward at the
    // 0.5 the old coercion installed for `[]` publishes 0.9820, and at the 2.0 it installed for
    // `[2]` publishes 0.7311. One bad field, and the answer is over- or under-confident depending on
    // which way `Number()` happened to round it.
    const { agent } = loadWith([[2], 2.0, 3.0]);
    const r: any = await agent.systemOne("hi", CHOICE);
    expect(r.answers.d.probabilities.x).toBeCloseTo(0.8808, 4);
  });

  it("leaves a list of three real numbers alone", () => {
    const { agent, warnings } = loadWith([1.0, 2.0, 0.5]);
    expect(agent.temperature).toEqual([1.0, 2.0, 0.5]);
    expect(warnings).toEqual([]);
  });
});
