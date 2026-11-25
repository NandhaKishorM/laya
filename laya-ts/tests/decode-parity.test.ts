import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { Agent } from "../src/agent.js";

// `laya-java/fixtures/decode.json` is the shared decode oracle: `gen_fixtures.py:1845-1860` hands
// each synthesised row to laya's own `Agent._decode_answers` and records the answer, so the same
// file is what the Java port asserts against field-by-field at tolerance 0.0
// (`laya-java/.../decode/DecoderParityTest.java`). The rows hit branches rather than look
// realistic -- every temperature bucket, a top-2 tie, an exactly uniform 20-option softmax, +/-700
// magnitudes that overflow a naive `exp`, a logits block three columns wider than the option
// count, and a per-language override -- which is the point: none of it needs a checkpoint.
//
// `option_order` is out of scope here. The port has no un-permutation step in the decode layer and
// no permutation step in the prompt layer, so the two permuted cases per checkpoint
// (`choice-4-permuted`, `score-permuted`) cannot agree on `choice`/`probabilities`/`score`. Their
// permutation-invariant fields are still asserted below, so the rest of the row stays covered.
const fixture = JSON.parse(
  readFileSync(new URL("../../laya-java/fixtures/decode.json", import.meta.url), "utf8"),
) as Record<string, DecodeCheckpoint>;

type DecodeCase = {
  question: { t: "choice" | "score" | "noul"; ins: string; crit?: unknown; labels?: unknown };
  option_order: number[] | null;
  lang: string | null;
  logits: number[];
  logits_block_width: number;
  act: number[];
  answer: Record<string, unknown>;
};

type DecodeCheckpoint = {
  config: {
    temperature: number[];
    temperature_by_options: Record<string, number>;
    lang_temperatures_raw: Record<
      string,
      { temperature: number[]; temperature_by_options: Record<string, number> }
    >;
  };
  qtypes: Record<string, number>;
  cases: Record<string, DecodeCase>;
};

const fakeProvider = () => ({
  async runEncoder(_b: unknown) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: unknown) { return { logits: [[2, 0]], act: [[3, 0]] }; },
});

/** The Agent the reference's `shim` is: the checkpoint's clamped scales and one language override. */
function agentFor(ck: DecodeCheckpoint): Agent {
  return new Agent({
    provider: fakeProvider(),
    cfg: {
      temperature: ck.config.temperature,
      temperature_by_options: ck.config.temperature_by_options,
    },
    lang_temperatures: ck.config.lang_temperatures_raw,
  } as never);
}

/** The row as the head returns it: padded past the option count with large finite values. */
function paddedRow(c: DecodeCase): number[] {
  const row = [...c.logits];
  while (row.length < c.logits_block_width) row.push(99.0);
  return row;
}

function decodeRow(agent: Agent, c: DecodeCase): Record<string, unknown> {
  const decodeRow = (agent as unknown as {
    _decodeRow(
      logits: number[],
      act: number[],
      q: DecodeCase["question"],
      k: number,
      lang: string | null,
    ): Record<string, unknown>;
  })._decodeRow;
  return decodeRow.call(agent, paddedRow(c), c.act, c.question, c.logits.length, c.lang);
}

describe("decode.json parity", () => {
  for (const [checkpointName, ck] of Object.entries(fixture)) {
    if ("skipped" in (ck as unknown as Record<string, unknown>)) continue;
    const agent = agentFor(ck);
    describe(checkpointName, () => {
      for (const [caseName, c] of Object.entries(ck.cases)) {
        if (c.option_order !== null) continue;
        it(`${caseName} decodes to the reference answer`, () => {
          expect(decodeRow(agent, c)).toEqual(c.answer);
        });
      }
    });
  }
});

describe("decode.json parity across option_order", () => {
  for (const [checkpointName, ck] of Object.entries(fixture)) {
    if ("skipped" in (ck as unknown as Record<string, unknown>)) continue;
    const agent = agentFor(ck);
    for (const [caseName, c] of Object.entries(ck.cases)) {
      if (c.option_order === null) continue;
      it(`${checkpointName}/${caseName} keeps the permutation-invariant fields`, () => {
        const got = decodeRow(agent, c);
        expect(got.type).toBe(c.answer.type);
        expect(got.confidence).toBe(c.answer.confidence);
        expect(got.answer_confidence).toBe(c.answer.answer_confidence);
        expect(got.action).toEqual(c.answer.action);
      });
    }
  }
});

describe("the decode layer reads the action row as a probability", () => {
  // `laya/agent.py:1329` (`_forward` returns `torch.softmax(act, -1)`) and `laya/onnx_agent.py:805`
  // normalise the action head at the model boundary; `Agent._decode_answers` then reports
  // `round(float(act[r, 0]), 4)` (`agent.py:1365`), and the Java decoder takes the pair as
  // `actionProbabilities` (`DecoderParityTest.java:71-72`). Normalising inside the decode layer
  // instead re-normalised the row: softmax([0.9, 0.1])[0] = 0.6899.
  const ck = fixture.english;
  const agent = agentFor(ck);
  const caseOf = (act: number[]): DecodeCase => ({
    ...ck.cases["choice-2"],
    act,
  });

  it("reports the first entry of the row the reference reports", () => {
    expect(decodeRow(agent, caseOf([0.9, 0.1])).action).toEqual({ act_probability: 0.9 });
    expect(decodeRow(agent, caseOf([0.99, 0.01])).action).toEqual({ act_probability: 0.99 });
    expect(decodeRow(agent, caseOf([0.25, -0.5])).action).toEqual({ act_probability: 0.25 });
  });

  it("rounds the reported value to four decimals, as round(_, 4) does", () => {
    expect(decodeRow(agent, caseOf([0.123456, 0.876544])).action).toEqual({
      act_probability: 0.1235,
    });
  });

  it("predict() normalises the provider's action logits once, at the model boundary", async () => {
    const provider = {
      async runEncoder(_b: unknown) { return { lastHidden: [[[1, 0]], [[0, 1]]] }; },
      async runHead(_h: unknown) {
        return { logits: [[2.0, 0.0]], act: [[3.0, 0.0]] };
      },
    };
    const a = new Agent({ provider } as never);
    const r = await a.systemOne("s", {
      q: { type: "choice", instructions: "i", criteria: { a: "1", b: "2" } },
    } as never);
    // softmax([3, 0])[0] = 0.95257…, one normalisation of the row the model exports as `act_logits`.
    expect(r.answers.q.action).toEqual({ act_probability: 0.9526 });
  });
});
