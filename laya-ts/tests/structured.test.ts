import { describe, expect, it } from "vitest";
import {
  answersToJson,
  decide,
  planFromJsonSchema,
  questionsFromJsonSchema,
  SchemaError,
  MAX_OPTIONS,
  MAX_PROPERTIES,
  MAX_SCORE_LEVELS,
} from "../src/structured.js";
import { Agent } from "../src/agent.js";
import { Router } from "../src/router.js";

// Mirrors tests/test_structured.py on the Python side.
const SCHEMA = {
  type: "object",
  properties: {
    department: { type: "string", enum: ["billing", "support", "sales"], description: "Which team?" },
    urgency: { type: "integer", minimum: 0, maximum: 2 },
    needs_human: { type: "boolean" },
    priority: { enum: [1, 2, 3] },
  },
};

const ANSWERS = {
  department: { type: "choice", choice: "billing", confidence: 0.9,
    probabilities: { billing: 0.9, support: 0.1, sales: 0.0 } },
  urgency: { type: "score", score: 1.2, confidence: 0.5,
    probabilities: { "0": 0.1, "1": 0.2, "2": 0.7 }, legend: {} },
  needs_human: { type: "noul", noul: 0.8, confidence: 0.8 },
  priority: { type: "choice", choice: "2", confidence: 0.7,
    probabilities: { "1": 0.2, "2": 0.7, "3": 0.1 } },
};

describe("structured/mapping", () => {
  const questions = questionsFromJsonSchema(SCHEMA);
  it("enum is a choice, description becomes instructions", () => {
    expect(questions.department.type).toBe("choice");
    expect(questions.department.instructions).toBe("Which team?");
    expect(Object.keys(questions.department.criteria as object)).toEqual(["billing", "support", "sales"]);
  });
  it("bounded number is a score with level labels", () => {
    expect(questions.urgency.type).toBe("score");
    expect(questions.urgency.criteria).toEqual(["0", "1", "2"]);
    expect(questions.urgency.instructions).toBe("Score `urgency` from 0 to 2");
  });
  it("boolean is noul", () => {
    expect(questions.needs_human.type).toBe("noul");
    expect(questions.needs_human.instructions).toBe("Is `needs_human` true?");
  });
  it("integer enum becomes a choice with string labels", () => {
    expect(questions.priority.type).toBe("choice");
    expect(Object.keys(questions.priority.criteria as object)).toEqual(["1", "2", "3"]);
  });
  it("null enum member gets the 'null' label", () => {
    const q = questionsFromJsonSchema({
      type: "object", properties: { x: { enum: ["a", null] } },
    });
    expect(Object.keys(q.x.criteria as object)).toEqual(["a", "null"]);
  });
  it("const is a single-option choice", () => {
    const q = questionsFromJsonSchema({ type: "object", properties: { x: { const: "billing" } } });
    expect(q.x.type).toBe("choice");
    expect(Object.keys(q.x.criteria as object)).toEqual(["billing"]);
  });
  it("all-boolean enum is noul", () => {
    const q = questionsFromJsonSchema({ type: "object", properties: { x: { enum: [true, false] } } });
    expect(q.x.type).toBe("noul");
  });
  it("nullable boolean remains noul", () => {
    const q = questionsFromJsonSchema({ type: "object", properties: { x: { type: ["null", "boolean"] } } });
    expect(q.x.type).toBe("noul");
  });
  it("plan has one field per property", () => {
    expect(planFromJsonSchema(SCHEMA)).toHaveLength(4);
  });
});

describe("structured/projection", () => {
  it("projects answers onto schema values", () => {
    const values = answersToJson(ANSWERS, SCHEMA);
    expect(values.department).toBe("billing");
    expect(values.urgency).toBe(2); // argmax level, not the raw 1.2 score
    expect(values.needs_human).toBe(true);
    expect(values.priority).toBe(2); // integer enum keeps its type
    expect(typeof values.priority).toBe("number");
  });
  it("false noul stays false", () => {
    const v = answersToJson({ x: { type: "noul", noul: 0.2 } },
      { type: "object", properties: { x: { type: "boolean" } } });
    expect(v.x).toBe(false);
  });
  it("score without probabilities falls back to the rounded score", () => {
    const v = answersToJson({ x: { type: "score", score: 1.6, confidence: 0.5 } },
      { type: "object", properties: { x: { type: "integer", minimum: 0, maximum: 2 } } });
    expect(v.x).toBe(2);
  });
  it("null choice value projects back to null", () => {
    const v = answersToJson({ x: { type: "choice", choice: "null" } },
      { type: "object", properties: { x: { enum: ["a", null] } } });
    expect(v.x).toBe(null);
  });
});

describe("structured/rejections", () => {
  const bad = (schema: unknown) => () => questionsFromJsonSchema(schema);
  it("rejects enum values with the same choice label", () => {
    for (const values of [[1, "1"], [null, "null"], [true, "true"]]) {
      expect(bad({ type: "object", properties: { x: { enum: values } } }))
        .toThrowError(/properties\.x: enum values produce duplicate choice labels/);
    }
  });
  it("rejects a free string", () => {
    expect(bad({ type: "object", properties: { a: { type: "string" } } }))
      .toThrowError(/properties\.a: a free string cannot be a fixed option set/);
  });
  it("rejects multiple non-null types even in a nullable union", () => {
    for (const types of [["boolean", "integer"], ["null", "integer", "boolean"]]) {
      expect(bad({ type: "object", properties: { a: { type: types, minimum: 0, maximum: 2 } } }))
        .toThrowError(/properties\.a: .*multiple non-null types/);
    }
  });
  it("rejects arrays, nested objects and $ref", () => {
    expect(bad({ type: "object", properties: { a: { type: "array", items: { type: "string" } } } }))
      .toThrowError(SchemaError);
    expect(bad({ type: "object", properties: { a: { type: "object", properties: {} } } }))
      .toThrowError(/nested objects are not supported/);
    expect(bad({ type: "object", properties: { a: { $ref: "#/$defs/X" } } }))
      .toThrowError(/\$ref\/recursion is not supported/);
  });
  it("rejects an unbounded or too-wide number", () => {
    expect(bad({ type: "object", properties: { a: { type: "integer", minimum: 0 } } }))
      .toThrowError(/needs integer 'minimum' and 'maximum'/);
    expect(bad({ type: "object", properties: { a: { type: "integer", minimum: 0, maximum: 100 } } }))
      .toThrowError(new RegExp(`exceeds MAX_SCORE_LEVELS=${MAX_SCORE_LEVELS}`));
  });
  it("rejects too many properties or options", () => {
    const manyProps = Object.fromEntries(
      Array.from({ length: MAX_PROPERTIES + 1 }, (_, i) => [`p${i}`, { type: "boolean" }]));
    expect(bad({ type: "object", properties: manyProps }))
      .toThrowError(new RegExp(`exceeds MAX_PROPERTIES=${MAX_PROPERTIES}`));
    const manyOpts = Array.from({ length: MAX_OPTIONS + 1 }, (_, i) => `v${i}`);
    expect(bad({ type: "object", properties: { a: { type: "string", enum: manyOpts } } }))
      .toThrowError(new RegExp(`exceeds MAX_OPTIONS=${MAX_OPTIONS}`));
  });
  it("rejects a non-object root and empty properties", () => {
    expect(bad({ type: "array" })).toThrowError(/top level must be an object with 'properties'/);
    expect(bad({ type: "object", properties: {} })).toThrowError(/non-empty object/);
    expect(bad("nope")).toThrowError(/expected a JSON schema object, got str/);
  });
});

describe("structured/decide", () => {
  class FakeRunner {
    calls: { state: unknown; questions: unknown; opts: unknown }[] = [];
    constructor(private answers: unknown) {}
    async predict(state: unknown, questions: any, opts: any = {}) {
      this.calls.push({ state, questions, opts });
      return { answers: this.answers, usage: { input_tokens: 1, output_tokens: 0 },
        routing: { model: "english" } };
    }
  }

  it("returns schema-shaped values and forwards state + kwargs", async () => {
    const runner = new FakeRunner(ANSWERS);
    const out: any = await decide(runner, "some state", SCHEMA, { hooksRaise: false });
    expect(out.department).toBe("billing");
    expect(runner.calls[0].state).toBe("some state");
    expect((runner.calls[0].questions as any).department.type).toBe("choice");
    expect((runner.calls[0].opts as any).hooksRaise).toBe(false);
  });

  it("returnDetails exposes confidence, probabilities, usage and routing", async () => {
    const runner = new FakeRunner(ANSWERS);
    const d = await decide(runner, "some state", SCHEMA, { returnDetails: true });
    expect(d.confidence.department).toBe(0.9);
    expect(d.probabilities.needs_human).toEqual({ false: 0.2, true: 0.8 });
    expect(d.usage).toEqual({ input_tokens: 1, output_tokens: 0 });
    expect(d.routing).toEqual({ model: "english" });
    expect(d.values.priority).toBe(2);
  });

  it("questions pass-through returns the raw answers", async () => {
    const raw = { a: { type: "noul", noul: 0.9, confidence: 0.9 } };
    const runner = new FakeRunner(raw);
    const out = await decide(runner, "s", undefined,
      { questions: { a: { type: "noul", instructions: "?" } } });
    expect(out).toEqual(raw);
  });

  it("requires exactly one of schema or questions", async () => {
    const runner = new FakeRunner({});
    await expect(decide(runner, "s")).rejects.toThrow(/exactly one of schema= or questions=/);
    await expect(decide(runner, "s", SCHEMA, { questions: {} }))
      .rejects.toThrow(/exactly one of schema= or questions=/);
  });

  it("accepts a model exposing toJSONSchema()", async () => {
    const runner = new FakeRunner(ANSWERS);
    const out: any = await decide(runner, "s", { toJSONSchema: () => SCHEMA });
    expect(out.department).toBe("billing");
  });
});

describe("structured/methods", () => {
  const fakeProvider = () => ({
    async runEncoder(b: any) {
      const n = b?.inputIds?.length ?? 2;
      return { lastHidden: Array.from({ length: n }, (_, i) => [i + 1, 0]) };
    },
    async runHead(h: any) {
      const n = h?.length ?? 1;
      return {
        logits: Array.from({ length: n }, (_, i) => (i % 2 === 0 ? [2, 0] : [0, 2])),
        act: Array.from({ length: n }, () => [3, 0]),
      };
    },
  });

  it("Agent.decide runs end to end against a stub provider", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const values = await a.decide("hi", {
      type: "object",
      properties: {
        department: { type: "string", enum: ["billing", "support"] },
        needs_human: { type: "boolean" },
      },
    });
    expect(values.department).toBe("billing"); // argmax of the fake logits
    expect(values.needs_human).toBe(true); // fake act head strongly positive
  });

  it("Router exposes decide", () => {
    expect(typeof (Router.prototype as any).decide).toBe("function");
  });
});

