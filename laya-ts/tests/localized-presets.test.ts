import { describe, expect, it } from "vitest";
import {
  emailQuestions, guardQuestions, moderationQuestions, routerQuestions,
  triageQuestions, presetLabels, triageLabels,
} from "../src/index.js";
import type { PresetName } from "../src/index.js";

const builders = {
  email: (language: string) => emailQuestions(undefined, language),
  guard: guardQuestions,
  moderation: moderationQuestions,
  router: routerQuestions,
  triage: triageQuestions,
};

describe("localized presets through the public API", () => {
  for (const name of Object.keys(builders) as PresetName[]) {
    it(`${name} preserves answer schemas and provides independent labels`, () => {
      const english = builders[name]("en");
      const swedish = builders[name]("sv-SE");
      expect(swedish).toEqual(builders[name]("sv"));
      expect(Object.keys(swedish)).toEqual(Object.keys(english));
      const labels = presetLabels(name, "sv-SE");
      for (const id of Object.keys(english)) {
        const en = english[id] as { type: string; criteria?: unknown };
        const sv = swedish[id] as { type: string; criteria?: unknown };
        expect(sv.type).toBe(en.type);
        expect(labels[id]).toBeTruthy();
        if (Array.isArray(en.criteria)) {
          expect(sv.criteria).toHaveLength(en.criteria.length);
        } else if (en.criteria && typeof en.criteria === "object") {
          expect(Object.keys(sv.criteria as object)).toEqual(Object.keys(en.criteria));
        }
      }
      labels[Object.keys(labels)[0]] = "changed by caller";
      expect(presetLabels(name, "sv")).not.toEqual(labels);
      expect(() => builders[name]("de")).toThrow();
    });
  }
  it("preserves the original English urgency question without added criteria", () => {
    expect(triageQuestions().is_urgent).toEqual({
      type: "noul", instructions: "Does `message` communicate time pressure or a deadline?",
    });
  });
  it("exports the triage convenience API and router alias", () => {
    expect(triageLabels("sv")).toEqual(presetLabels("triage", "sv"));
    expect(triageLabels("sv").refund).toBe("Återbetalning");
    expect(presetLabels("model_router", "sv")).toEqual(presetLabels("router", "sv"));
  });
  it.each(["constructor", "toString", "__proto__", "unknown"])("rejects invalid preset %s", (name) => {
    expect(() => presetLabels(name as PresetName, "sv")).toThrow(TypeError);
  });
});
