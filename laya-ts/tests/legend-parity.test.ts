import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import { renderCriterion, renderOptions } from "../src/common.js";

/**
 * A `score` answer's `legend` maps a level index to the TEXT of that level, and its keys are
 * already strings. Its values used to be the caller's own JSON types, so a numeric scale came back
 * as `{"0": 1}`, a boolean as `{"0": true}` and a null as `{"0": null}` -- which a Jev client
 * refuses to parse (#302). The Python side was fixed the same way in `laya/agent.py` and
 * `laya/onnx_agent.py`, so all three runtimes have to agree on the value types.
 *
 * The Python numbers these are pinned against, produced by `render_criterion` in `laya/common.py`:
 *
 *   render_criterion(1)          -> '1'
 *   render_criterion(1.5)        -> '1.5'
 *   render_criterion(True)       -> 'true'
 *   render_criterion(None)       -> 'null'
 *   render_criterion({'a': 1})   -> '{"a": 1}'
 *   render_criterion([1, 2])     -> '[1, 2]'
 */

const fakeProvider = (n: number) => ({
  async runEncoder(_b: any) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: any) {
    const row = Array.from({ length: n }, (_, i) => (i === 0 ? 2 : 0));
    return { logits: [row], act: [[3, 0]] };
  },
});

/** The legend for one score question, through the public API. */
async function legendFor(criteria: unknown[]): Promise<Record<string, unknown>> {
  const agent = new Agent({ provider: fakeProvider(criteria.length) } as any);
  const res: any = await agent.systemOne("s", {
    q: { type: "score", instructions: "q?", criteria },
  });
  return res.answers.q.legend;
}

describe("score legend value types", () => {
  it("renders numeric levels as strings, not numbers", async () => {
    const legend = await legendFor([1, 2, 3]);
    expect(legend).toEqual({ "0": "1", "1": "2", "2": "3" });
    for (const v of Object.values(legend)) expect(typeof v).toBe("string");
  });

  it("renders float levels as strings", async () => {
    expect(await legendFor([1.5, 2.5])).toEqual({ "0": "1.5", "1": "2.5" });
  });

  it("renders boolean levels with JSON spelling, matching Python", async () => {
    // Python's `render_criterion(True)` is 'true', not 'True' -- this is the cross-runtime pin
    expect(await legendFor([true, false])).toEqual({ "0": "true", "1": "false" });
  });

  it("rejects a null level rather than rendering it, matching the Python guard", async () => {
    // A null score level never reaches the legend: both runtimes refuse it up front, so
    // `{"0": null}` (#302) cannot be produced on either. This pins the TS half of that guard.
    await expect(legendFor([null])).rejects.toThrow(/score level 0 is null/);
    await expect(legendFor(["low", null])).rejects.toThrow(/score level 1 is null/);
  });

  it("renders a structured level as the JSON the model was shown, not a repr", async () => {
    expect(await legendFor([{ a: 1 }])).toEqual({ "0": '{"a": 1}' });
    expect(await legendFor([[1, 2]])).toEqual({ "0": "[1, 2]" });
  });

  it("keeps string levels unchanged", async () => {
    expect(await legendFor(["low", "high"])).toEqual({ "0": "low", "1": "high" });
  });

  it("agrees with renderCriterion, which is what the option text already used", async () => {
    // the point of the fix: the legend value and the option text are the same rendering.
    // `null` is absent because it is rejected before this point (see the test above).
    for (const level of [1, 1.5, true, false, "low"]) {
      const legend = await legendFor([level]);
      const optionText = renderOptions({ t: "score", ins: "q?", crit: [level] } as any)[0];
      expect(legend["0"]).toBe(renderCriterion(level));
      expect(optionText).toBe(`level 0: ${renderCriterion(level)}`);
    }
  });

  it("does not change the probabilities keys or the score itself", async () => {
    const agent = new Agent({ provider: fakeProvider(3) } as any);
    const res: any = await agent.systemOne("s", {
      q: { type: "score", instructions: "q?", criteria: [1, 2, 3] },
    });
    // the legend is stringified, the probability keys were already strings, and the score is a
    // number either way -- so only the legend changed
    expect(Object.keys(res.answers.q.probabilities)).toEqual(["0", "1", "2"]);
    expect(typeof res.answers.q.score).toBe("number");
    expect(typeof res.answers.q.confidence).toBe("number");
  });
});
