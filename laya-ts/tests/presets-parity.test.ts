import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  emailQuestions,
  guardQuestions,
  moderationQuestions,
  routerQuestions,
  triageQuestions,
} from "../src/presets.js";

// `laya-java/fixtures/presets.json` is the shared preset oracle: the Java port reads every entry
// out of it, and `laya-java/scripts/gen_fixtures.py` regenerates it by calling `laya/presets.py`, so
// it is the reference's own output rather than a transcription of it. The TypeScript port had no
// test that consulted it, which is how `emailQuestions({})` came to disagree with the reference:
// Python's `categories or {…defaults}` substitutes on *truthiness*, so an empty map gets the six
// default categories -- the fixture entry `email_questions_empty_categories` is literally
// `mod.email_questions({})` -- while the port's `categories ?? {…defaults}` substitutes only on
// null and undefined and so builds a choice question with no options to choose from.
//
// The fixture also carries `state_field`, `state_field_hostile` and `hostile_specs` entries for
// `laya.presets.state_field`, which this port has no counterpart for. Those are out of scope here
// and deliberately not asserted.
const fixture = JSON.parse(
  readFileSync(new URL("../../laya-java/fixtures/presets.json", import.meta.url), "utf8"),
) as Record<string, Record<string, unknown>>;

const BUILDERS: Record<string, () => Record<string, unknown>> = {
  triage_questions: triageQuestions,
  guard_questions: guardQuestions,
  moderation_questions: moderationQuestions,
  router_questions: routerQuestions,
  email_questions: () => emailQuestions(),
};

describe("presets against the shared Java fixture", () => {
  for (const [name, build] of Object.entries(BUILDERS)) {
    it(`${name} matches the committed oracle`, () => {
      expect(build()).toEqual(fixture[name]);
    });
  }

  it("email_questions uses a non-empty category map verbatim", () => {
    const cats = JSON.parse(
      JSON.stringify({
        ops: "incidents and deploys",
        legal: "contracts and compliance",
      }),
    ) as Record<string, string>;
    expect(emailQuestions(cats)).toEqual(fixture.email_questions_custom);
    expect((emailQuestions(cats).category as { criteria: Record<string, string> }).criteria).toEqual(cats);
  });

  it("email_questions falls back to the defaults on null and undefined", () => {
    expect(emailQuestions(undefined)).toEqual(fixture.email_questions);
    expect(emailQuestions(null as unknown as Record<string, string>)).toEqual(fixture.email_questions);
  });

  it("email_questions falls back to the defaults on an empty map, as Python's `or` does", () => {
    expect(emailQuestions({})).toEqual(fixture.email_questions_empty_categories);
    const category = emailQuestions({}).category as { criteria: Record<string, string> };
    expect(Object.keys(category.criteria)).toHaveLength(6);
  });

  it("every form the reference's `or` treats as empty builds the default question set", () => {
    // Measured against `laya.presets.email_questions(v)` for each of these: `None`, `{}`, `""`,
    // `0`, `False` and `[]` all substitute to the defaults, and only a non-empty mapping is used.
    const empties: unknown[] = [{}, "", 0, false, [], undefined, null, 0.0];
    for (const empty of empties) {
      expect(emailQuestions(empty as Record<string, string>), JSON.stringify(empty)).toEqual(
        fixture.email_questions_empty_categories,
      );
    }
  });
});
