// laya-ts/tests/option-order.test.ts — a question's `option_order` reaches both halves of the
// prediction: the prompt presents option `order[s]` in slot `s`, and the decoded row is put back
// into the caller's option order before an answer is chosen.
//
// The expected answers are not hand-computed. They are printed by `/tmp/probe_option_order.py`,
// which calls laya's own `Agent._decode_answers` through the same `Agent.__new__` shim
// `laya-java/scripts/gen_fixtures.py:1838-1860` uses to build `decode.json` -- clamped temperature
// tables, an `[0.25, -0.5]` action row, a logits block three columns wider than the option count --
// so the numbers below are the reference decoder's output for these rows, not a restatement of this
// port's arithmetic.
import { describe, expect, it, vi } from "vitest";
import { Agent, checkQuestion, toInternal } from "../src/agent.js";
import {
  buildQuestionPrefix,
  renderOptions,
  unpermuteProbs,
  type InternalQ,
} from "../src/common.js";
import { predictShortlist } from "../src/shortlist.js";
import type { TokenizerLike } from "../src/tokenizer.js";

type QuestionDef = Record<string, unknown>;

const padded = (logits: number[]): number[] => [...logits, 99.0, 99.0, 99.0];

/** `Agent._decodeRow` at the same seam `decode-parity.test.ts` uses. */
function decodeRow(agent: Agent, q: InternalQ, logits: number[]): Record<string, unknown> {
  const row = (agent as unknown as {
    _decodeRow(
      logits: number[], act: number[], q: InternalQ, k: number, lang: string | null,
    ): Record<string, unknown>;
  })._decodeRow;
  return row.call(agent, padded(logits), [0.25, -0.5], q, logits.length, null);
}

const stubAgent = () => new Agent({
  provider: {
    async runEncoder() { return { lastHidden: [[1, 0]] }; },
    async runHead() { return { logits: [[2, 0]], act: [[1, 0]] }; },
  },
} as never);

describe("option_order validation", () => {
  const choice = (order: unknown): QuestionDef => ({
    type: "choice", instructions: "pick", criteria: { a: "1", b: "2", c: "3" }, option_order: order,
  });

  it("accepts every permutation of the option indices", () => {
    for (const order of [[0, 1, 2], [2, 0, 1], [1, 2, 0]]) {
      expect(() => checkQuestion("q", choice(order))).not.toThrow();
    }
  });

  it("refuses an order that is not one option per slot", () => {
    // Each arm is a different way an order loses or duplicates an option. All of them would reach
    // the encoder silently if the field were not validated here; Python refuses them at the same
    // point (`laya/agent.py:1088-1099`).
    const bad: unknown[] = [
      [0, 1], [0, 1, 2, 3], [0, 0, 1], [2, 0, 3], [1, 2], [], [0, 1, "2"], [0, 1, 1.5],
      ["a", "b", "c"], [true, false, true], 2, null, { 0: 1 },
    ];
    for (const order of bad) {
      expect(() => checkQuestion("q", choice(order))).toThrow(/permutation of range\(3\)/);
    }
  });

  it("names the question, the required range and the value it refused", () => {
    expect(() => checkQuestion("q", choice([3, 0, 0]))).toThrow(
      /question "q": 'option_order' must be a permutation of range\(3\) -- one slot per option, each option once -- got \[3,0,0\]/,
    );
  });

  it("sizes a score by its levels and a noul by its pair", () => {
    expect(() => checkQuestion("q", {
      type: "score", instructions: "rate", criteria: ["bad", "ok", "good"], option_order: [2, 0, 1],
    })).not.toThrow();
    expect(() => checkQuestion("q", {
      type: "score", instructions: "rate", criteria: ["bad", "ok", "good"], option_order: [1, 0],
    })).toThrow(/permutation of range\(3\)/);
    expect(() => checkQuestion("q", {
      type: "noul", instructions: "holds", option_order: [1, 0],
    })).not.toThrow();
    expect(() => checkQuestion("q", {
      type: "noul", instructions: "holds", option_order: [0, 1, 2],
    })).toThrow(/permutation of range\(2\)/);
  });

  it("counts a criteria list as one option per entry", () => {
    expect(() => checkQuestion("q", {
      type: "choice", instructions: "pick", criteria: ["a", "b"], option_order: [1, 0],
    })).not.toThrow();
    expect(() => checkQuestion("q", {
      type: "choice", instructions: "pick", criteria: ["a", "b"], option_order: [1, 0, 0],
    })).toThrow(/permutation of range\(2\)/);
  });
});

describe("toInternal carries the order", () => {
  it("sets optionOrder from the definition", () => {
    expect(toInternal({
      type: "choice", instructions: "pick", criteria: { a: "1", b: "2" }, option_order: [1, 0],
    } as never).optionOrder).toEqual([1, 0]);
  });

  it("leaves it unset when the question has none", () => {
    const q = toInternal({ type: "choice", instructions: "pick", criteria: { a: "1" } } as never);
    expect(Object.hasOwn(q, "optionOrder")).toBe(false);
  });

  it("truncates a numeric index the way int() does", () => {
    // Validation is what normally guarantees integral entries; the reference still writes `int(i)`
    // (`laya/agent.py:1129`) onto the internal question, so the port cannot hold a fractional index.
    expect(toInternal({
      type: "choice", instructions: "pick", criteria: { a: "1", b: "2", c: "3" },
      option_order: [2.7, 0.1, 1.9],
    } as never).optionOrder).toEqual([2, 0, 1]);
  });
});

describe("unpermuteProbs", () => {
  it("maps slot s back onto option order[s]", () => {
    expect(unpermuteProbs([0.1, 0.6, 0.3], [1, 2, 0])).toEqual([0.3, 0.1, 0.6]);
  });

  it("is the inverse of presenting the options in that order", () => {
    const opts = renderOptions({ t: "choice", ins: "i", crit: { a: "1", b: "2", c: "3" } });
    const order = [2, 0, 1];
    // Slot s carries option `order[s]`, so inverting the slot row has to land every value on its own
    // option -- asserted against the rendered option texts, not against another formula.
    const back = unpermuteProbs([0.7, 0.2, 0.1], order);
    expect(back.map((v, i) => [opts[i], v])).toEqual([["a: 1", 0.2], ["b: 2", 0.1], ["c: 3", 0.7]]);
  });

  it("returns the row untouched without an order or when the lengths disagree", () => {
    // `laya/common.py:567` leaves the row alone for None and for a length mismatch. The mismatch is
    // the arm that matters: a head row can be shorter than the order once options are lost to the
    // token budget, and re-indexing it would move probabilities onto options never shown.
    expect(unpermuteProbs([0.6, 0.4])).toEqual([0.6, 0.4]);
    expect(unpermuteProbs([0.6, 0.4], null)).toEqual([0.6, 0.4]);
    expect(unpermuteProbs([0.6, 0.4], [0, 1, 2])).toEqual([0.6, 0.4]);
  });
});

/** One id per encoded string, remembered so a slot's token traces back to the option it shows. */
type IdentityTok = TokenizerLike & { idOf(text: string): number };

function identityTokenizer(): IdentityTok {
  const ids = new Map<string, number>();
  const tok: IdentityTok = {
    clsId: 101,
    sepId: 102,
    maskId: 103,
    padId: 0,
    maskToken: "[MASK]",
    encode(text: string): number[] {
      const key = text.trim();
      if (!ids.has(key)) ids.set(key, 1000 + ids.size);
      return [ids.get(key)!];
    },
    idOf(text: string): number {
      return ids.get(text.trim()) ?? -1;
    },
  };
  return tok;
}

// Criteria written as bare labels: `renderOptions` drops an empty description, so the text the
// model sees for option `i` is exactly the key, which is what makes a slot traceable.
const THREE = { billing: "", refund: "", technical: "" };

describe("the prompt presents options in the caller's order", () => {
  const q: InternalQ = { t: "choice", ins: "i", crit: THREE };
  const opts = renderOptions(q);

  it("puts option order[s] in slot s", () => {
    for (const order of [[0, 1, 2], [2, 0, 1], [1, 2, 0]]) {
      const tok = identityTokenizer();
      const built = buildQuestionPrefix(tok, q, 512, 192, order);
      expect(built.markers).toHaveLength(3);
      // A slot's chunk is [maskId, ...option tokens], so the id right after a marker is the option.
      order.forEach((optIdx, s) => {
        expect(built.ids[built.markers[s] + 1]).toBe(tok.idOf(opts[optIdx]));
      });
    }
  });

  it("shows the same options the answer keys will name", () => {
    const tok = identityTokenizer();
    const built = buildQuestionPrefix(tok, q, 512, 192, [2, 0, 1]);
    expect(built.markers.map((m) => built.ids[m + 1])).toEqual(
      ["technical", "billing", "refund"].map((o) => tok.idOf(o)),
    );
    // Without the prompt half -- order ignored -- the same row would be decoded against the wrong
    // options, which is the silent failure this field exists to prevent.
    const canonical = buildQuestionPrefix(tok, q, 512, 192);
    expect(canonical.ids[canonical.markers[0] + 1]).toBe(tok.idOf("billing"));
  });
});

describe("the decoded row answers in the caller's option order", () => {
  const a = stubAgent();

  it("matches the reference on a rotated choice", () => {
    expect(decodeRow(a, {
      t: "choice", ins: "i", crit: { a: "1", b: "2", c: "3" }, optionOrder: [1, 2, 0],
    }, [0.4, 2.2, 0.9])).toEqual({
      type: "choice", choice: "c",
      probabilities: { a: 0.1895, b: 0.115, c: 0.6955 },
      confidence: 0.2568, answer_confidence: 0.6955,
      action: { act_probability: 0.25 },
    });
  });

  it("matches the reference on the same row read in canonical order", () => {
    // The pair is the point: the logits are unchanged, only the slots differ, so the answer moves
    // from `c` to `b` because the order says which option each slot is showing.
    expect(decodeRow(a, {
      t: "choice", ins: "i", crit: { a: "1", b: "2", c: "3" }, optionOrder: [0, 1, 2],
    }, [0.4, 2.2, 0.9])).toEqual({
      type: "choice", choice: "b",
      probabilities: { a: 0.115, b: 0.6955, c: 0.1895 },
      confidence: 0.2568, answer_confidence: 0.6955,
      action: { act_probability: 0.25 },
    });
  });

  it("matches the reference on a reversed four-option row", () => {
    expect(decodeRow(a, {
      t: "choice", ins: "i", crit: { a: "1", b: "2", c: "3", d: "4" }, optionOrder: [3, 2, 1, 0],
    }, [0.5, 2.5, -1.0, 2.5])).toEqual({
      type: "choice", choice: "a",
      probabilities: { a: 0.4618, b: 0.0139, c: 0.4618, d: 0.0625 },
      confidence: 0.3173, answer_confidence: 0.4618,
      action: { act_probability: 0.25 },
    });
  });

  it("matches the reference on a permuted score, where the level index is the value", () => {
    expect(decodeRow(a, {
      t: "score", ins: "i", crit: ["l0", "l1", "l2", "l3"], optionOrder: [2, 0, 3, 1],
    }, [0.2, 1.4, 0.9, -0.3])).toEqual({
      type: "score", score: 1.246,
      legend: { "0": "l0", "1": "l1", "2": "l2", "3": "l3" },
      probabilities: { "0": 0.4784, "1": 0.0874, "2": 0.1441, "3": 0.2901 },
      confidence: 0.1316, answer_confidence: 0.4784,
      action: { act_probability: 0.25 },
    });
  });

  it("matches the reference on a flipped noul, where slot 0 is showing the true option", () => {
    // Without the inversion `p[1]` is whatever the second slot happened to show, so a flipped noul
    // reports the complement of the reference: 0.832 where laya reports 0.168.
    expect(decodeRow(a, { t: "noul", ins: "i", crit: null, optionOrder: [1, 0] }, [0.3, 1.9])).toEqual({
      type: "noul", noul: 0.168, confidence: 0.832, answer_confidence: 0.832,
      action: { act_probability: 0.25 },
    });
    expect(decodeRow(a, { t: "noul", ins: "i", crit: null, optionOrder: [0, 1] }, [0.3, 1.9])).toEqual({
      type: "noul", noul: 0.832, confidence: 0.832, answer_confidence: 0.832,
      action: { act_probability: 0.25 },
    });
  });

  it("leaves a row shorter than the order alone, as the reference does", () => {
    // Options lost to the head budget make the row shorter than `option_order`; re-indexing by the
    // order would name an option that has no span on screen.
    expect(decodeRow(a, {
      t: "choice", ins: "i", crit: { a: "1", b: "2" }, optionOrder: [0, 1, 2],
    }, [2.0, 0.1])).toEqual({
      type: "choice", choice: "a", probabilities: { a: 0.8699, b: 0.1301 },
      confidence: 0.4423, answer_confidence: 0.8699,
      action: { act_probability: 0.25 },
    });
  });

  it("does not move an answer for a question with no order", () => {
    const plain = { t: "choice", ins: "i", crit: { a: "1", b: "2", c: "3" } } as InternalQ;
    expect(decodeRow(a, plain, [0.4, 2.2, 0.9]))
      .toEqual(decodeRow(a, { ...plain, optionOrder: undefined }, [0.4, 2.2, 0.9]));
  });
});

describe("predict() honours option_order end to end", () => {
  it("shows the permuted options and answers in the caller's order", async () => {
    let seen: { inputIds: number[][]; markerPos: number[][] } | null = null;
    const tok = identityTokenizer();
    const provider = {
      async runEncoder(batch: { inputIds: number[][]; markerPos: number[][] }) {
        seen = { inputIds: batch.inputIds, markerPos: batch.markerPos };
        return { lastHidden: batch.inputIds.map((row: number[]) => row.map(() => [1, 0])) };
      },
      // The reference arm's row: under this order slot 0 shows `b`, slot 1 shows `c`, slot 2 `a`.
      async runHead() { return { logits: [[0.4, 2.2, 0.9, 99.0, 99.0, 99.0]], act: [[1.0, 0.0]] }; },
    };
    const a = new Agent({ provider, tok } as never);
    const r = await a.systemOne("s", {
      q: {
        type: "choice", instructions: "i",
        criteria: { a: "", b: "", c: "" },
        option_order: [1, 2, 0],
      },
    } as never);

    // The prompt half, read back off the batch the model actually received.
    const { inputIds, markerPos } = seen as unknown as { inputIds: number[][]; markerPos: number[][] };
    expect(markerPos[0].map((m) => inputIds[0][m + 1])).toEqual(
      ["b", "c", "a"].map((o) => tok.idOf(o)),
    );
    // The answer half: the winning slot is reported against the option it was showing, and the
    // probabilities are keyed the way the reference keys them for this row and order.
    expect(r.answers.q.type).toBe("choice");
    const c = r.answers.q as typeof r.answers.q & { choice: string; probabilities: Record<string, number> };
    expect(c.choice).toBe("c");
    expect(c.probabilities).toEqual({ a: 0.1895, b: 0.115, c: 0.6955 });
  });

  it("refuses a bad order before the model is consulted", async () => {
    let called = false;
    const provider = {
      async runEncoder(_b: unknown) { called = true; return { lastHidden: [[1, 0]] }; },
      async runHead() { return { logits: [[1, 0]], act: [[1, 0]] }; },
    };
    const a = new Agent({ provider } as never);
    await expect(a.systemOne("s", {
      q: { type: "choice", instructions: "i", criteria: { a: "1", b: "2" }, option_order: [0, 2] },
    } as never)).rejects.toThrow(/permutation of range\(2\)/);
    expect(called).toBe(false);
  });

  it("changes nothing for a question that does not carry the field", async () => {
    const provider = {
      async runEncoder(b: { inputIds: number[][] }) {
        return { lastHidden: b.inputIds.map((row: number[]) => row.map(() => [1, 0])) };
      },
      async runHead() { return { logits: [[0.4, 2.2, 0.9, 99.0, 99.0, 99.0]], act: [[1.0, 0.0]] }; },
    };
    const questions = (order?: number[]): Record<string, QuestionDef> => ({
      q: {
        type: "choice", instructions: "i", criteria: { a: "", b: "", c: "" },
        ...(order === undefined ? {} : { option_order: order }),
      },
    });
    const identity = await new Agent({ provider, tok: identityTokenizer() } as never)
      .systemOne("s", questions([0, 1, 2]) as never);
    const none = await new Agent({ provider, tok: identityTokenizer() } as never)
      .systemOne("s", questions() as never);
    expect(identity.answers).toEqual(none.answers);
    expect(identity.usage).toEqual(none.usage);
    // And the rotated order does move it, so the equality above is not two copies of one behaviour.
    const rotated = await new Agent({ provider, tok: identityTokenizer() } as never)
      .systemOne("s", questions([1, 2, 0]) as never);
    expect(rotated.answers).not.toEqual(none.answers);
  });
});

describe("a narrowed question loses an order that no longer fits it", () => {
  // The drop already lived in `shortlist.ts`; the validation added here is what makes it
  // load-bearing, because a one-option shortlist still carrying a three-slot order would now be
  // refused outright. Python's remedy for the same case is the drop, not the refusal
  // (`laya/shortlist.py:426`).
  it("removes option_order from the narrowed definition and keeps the question valid", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    let reduced: Record<string, QuestionDef> | null = null;
    const stub = {
      async predict(_s: unknown, q: Record<string, QuestionDef>) {
        reduced = q;
        return { model: "m", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
      },
    } as never;
    const embed = async (texts: string[]) =>
      texts.map((t) => (t.includes("refund") ? [1, 0] : [0, 1]));
    await predictShortlist(stub, "billed twice refund", {
      d: {
        type: "choice", instructions: "pick",
        criteria: { card_arrival: "where is my card", transfer_fee: "fee on transfer", refund: "refund money" },
        option_order: [2, 0, 1],
      },
    }, embed as never, 1);

    const kept = reduced as unknown as Record<string, QuestionDef>;
    expect(Object.keys(kept.d["criteria"] as object)).toEqual(["refund"]);
    expect(Object.hasOwn(kept.d, "option_order")).toBe(false);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(() => checkQuestion("d", kept.d)).not.toThrow();
    // The counterfactual the drop exists to prevent: the same narrowed criteria still carrying the
    // original three-slot order is refused by the validation above.
    expect(() => checkQuestion("d", {
      type: "choice", instructions: "pick", criteria: { refund: "refund money" }, option_order: [2, 0, 1],
    })).toThrow(/permutation of range\(1\)/);
    warn.mockRestore();
  });
});
