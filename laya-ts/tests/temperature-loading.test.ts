import { describe, expect, it, vi } from "vitest";
import { Agent } from "../src/agent.js";

// Python pins this contract in `tests/test_temperature_loading.py`; the port carries the same
// load-time diagnostic (`src/agent.ts:448-458`) and had no test over it. `clampTemperature` refuses
// a bool -- `true` is not the number 1.0 and `false` is not the sharpening 0.0 -- but `Number(true)`
// is exactly the neutral 1.0 the clamp installs instead, so a `true` temperature from a checkpoint
// config was compared, matched, and loaded in silence. `false` was reported only because it floats
// to 0.0 and mismatches: the same value the clamp rejects, seen from two sides.

const fakeProvider = () => ({
  async runEncoder(_b: any) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: any) { return { logits: [[2, 0]], act: [[3, 0]] }; },
});

const CHOICE = { d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } } } as any;

function loadWith(temperature: unknown[], temperatureByOptions: Record<string, unknown> = {}) {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  try {
    const agent = new Agent({
      provider: fakeProvider(),
      cfg: { temperature, temperature_by_options: temperatureByOptions },
    } as any);
    return { agent, warnings: warn.mock.calls.map((call: unknown[]) => String(call[0])) };
  } finally {
    warn.mockRestore();
  }
}

describe("checkpoint temperature loading", () => {
  it.each([true, false])("reports a %s where a temperature belongs", async (value: boolean) => {
    const { agent, warnings } = loadWith([value, 2.0, 3.0], { "choice:2": value });
    expect(agent.temperature).toEqual([1.0, 2.0, 3.0]);
    expect(agent.temperatureByOptions).toEqual({ "choice:2": 1.0 });
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain(`temperature[0]=${value} -> 1`);
    expect(warnings[0]).toContain(`choice:2=${value} -> 1`);
    expect(warnings[0]).toContain("uncalibrated");
    // The neutral 1.0 is what the answers are actually scaled by: softmax([2, 0]) at T=1.
    const r: any = await agent.systemOne("hi", CHOICE);
    expect(r.answers.d.probabilities.x).toBeCloseTo(0.8808, 4);
  });

  it("keeps reporting an out-of-range value on every entry", () => {
    const { agent, warnings } = loadWith([0.1, 10, -1]);
    expect(agent.temperature).toEqual([0.5, 5.0, 0.5]);
    expect(warnings).toHaveLength(1);
    for (const name of ["temperature[0]", "temperature[1]", "temperature[2]"]) {
      expect(warnings[0]).toContain(name);
    }
  });

  it("does not report a number, or a numeric string that equals the applied value", () => {
    const { agent, warnings } = loadWith([0.5, "2", 5], { "choice:2": "1.5" });
    expect(agent.temperature).toEqual([0.5, 2.0, 5.0]);
    expect(agent.temperatureByOptions).toEqual({ "choice:2": 1.5 });
    expect(warnings).toEqual([]);
  });

  it("still reports a value that is not a number at all", () => {
    for (const value of [null, undefined, "", {}, "invalid"]) {
      const { agent, warnings } = loadWith([value, 2.0, 3.0]);
      expect(agent.temperature).toEqual([1.0, 2.0, 3.0]);
      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toContain("temperature[0]");
    }
    // A list is named too, and only the naming is asserted: `Number([])` is 0 so the port bounds it
    // to TEMP_MIN while Python's `float([])` raises and falls back to 1.0. Which of the two the
    // clamp installs is a separate question from whether the loader says anything about it.
    const empty = loadWith([[] as any, 2.0, 3.0]);
    expect(empty.warnings).toHaveLength(1);
    expect(empty.warnings[0]).toContain("temperature[0]");
  });
});
