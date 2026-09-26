import { describe, expect, it } from "vitest";
import { Router } from "../src/router.js";

// Parity with tests/test_blank_lang_routing.py: a blank or whitespace explicit `lang` names no
// language, so it must fall through to langGuess/detection instead of pinning multilingual.
const GENERIC = { intent: { type: "choice", instructions: "x", criteria: ["a", "b"] } };
// Detected as English, so "abstain" (english) and "forced multilingual" are distinguishable.
const ENGLISH = "I was charged twice for invoice 4411";
const GERMAN = "Mein Konto wurde zweimal belastet, bitte erstatten Sie";

describe("blank explicit lang", () => {
  const r = new Router();

  it("baseline: plain English routes english with no explicit reason", () => {
    const d = r.route(ENGLISH, GENERIC);
    expect(d.model).toBe("english");
    expect(d.reason).not.toContain("explicit lang=");
  });

  for (const lang of ["", "   ", null, undefined]) {
    it(`lang=${JSON.stringify(lang)} falls through to detection`, () => {
      const d = r.route(ENGLISH, GENERIC, { lang: lang as string | null });
      expect(d.model).toBe("english");
      expect(d.reason).not.toContain("explicit lang=");
      expect(d.detection).not.toBeNull();
    });
  }

  it("real codes still win", () => {
    expect(r.route(GERMAN, GENERIC, { lang: "en" }).model).toBe("english");
    expect(r.route(ENGLISH, GENERIC, { lang: "de" }).model).toBe("multilingual");
    expect(r.route(GERMAN, GENERIC, { lang: "en" }).reason).toContain("explicit lang=");
    expect(r.route(ENGLISH, GENERIC, { lang: "de" }).reason).toContain("explicit lang=");
  });

  for (const code of ["en", "EN", "en-US", "en_US", "en_US.UTF-8"]) {
    it(`code ${code} routes english`, () => {
      expect(r.route(ENGLISH, GENERIC, { lang: code }).model).toBe("english");
    });
  }

  for (const code of ["de", "fr", "zh_CN", "pt-BR"]) {
    it(`code ${code} routes multilingual`, () => {
      expect(r.route(ENGLISH, GENERIC, { lang: code }).model).toBe("multilingual");
    });
  }

  it("a blank lang does not mask an installed langGuess", () => {
    const hinted = new Router({ langGuess: "de" });
    const d = hinted.route(ENGLISH, GENERIC, { lang: "" });
    expect(d.model).toBe("multilingual");
    expect(d.reason).toContain("Router(lang_guess=...)");
  });
});
