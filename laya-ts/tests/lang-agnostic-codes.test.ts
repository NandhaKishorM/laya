import { describe, expect, it } from "vitest";
import { Router, englishFromCode } from "../src/router.js";

// Parity with the Python fix for #359: `C`, `POSIX` and `C.UTF-8` are valid `$LANG` values that
// name no language (`C.UTF-8` is the official Python image's default), and the ISO 639-2 special
// codes `und`/`zxx`/`mul` say the same. They abstain, so detection names the checkpoint instead
// of every request being pinned to the multilingual one.
const ENGLISH_STATE = "Please refund the duplicate charge on invoice 4411";

describe("lang codes that name no language", () => {
  const agnostic = ["C", "POSIX", "C.UTF-8", "c.utf8", "c", "posix", "und", "zxx", "mul", "UND", "Zxx", " und "];

  for (const code of agnostic) {
    it(`${JSON.stringify(code)} abstains and lets detection decide`, () => {
      expect(englishFromCode(code)).toBeNull();
      for (const opts of [{ lang: code }, { langGuess: code }]) {
        const d = new Router().route(ENGLISH_STATE, {}, opts);
        expect(d.model).toBe("english");
        expect(d.reason).not.toContain("explicit");
        expect(d.detection).not.toBeNull();
      }
      const installed = new Router({ langGuess: code }).route(ENGLISH_STATE, {});
      expect(installed.model).toBe("english");
      expect(installed.detection).not.toBeNull();
    });
  }

  it("blank codes still abstain", () => {
    expect(englishFromCode("")).toBeNull();
    expect(englishFromCode(null)).toBeNull();
    expect(englishFromCode("   ")).toBeNull();
  });

  for (const code of ["en", "eng", "english", "EN", "en-US", "en_US.UTF-8"]) {
    it(`${code} is still decisive English`, () => {
      expect(englishFromCode(code)).toBe(true);
      expect(new Router().route(ENGLISH_STATE, {}, { lang: code }).reason).toContain("explicit");
    });
  }

  for (const code of ["de", "fr", "zh", "ja", "pt-BR", "de_DE.UTF-8"]) {
    it(`${code} is still decisive non-English`, () => {
      expect(englishFromCode(code)).toBe(false);
      expect(new Router().route(ENGLISH_STATE, {}, { lang: code }).model).toBe("multilingual");
    });
  }

  it("the primary subtag is what is compared", () => {
    expect(englishFromCode("POSIX-1")).toBeNull();
    // `ca` (Catalan) and `mn` (Mongolian) share letters with agnostic codes but name languages.
    expect(englishFromCode("ca")).toBe(false);
    expect(englishFromCode("mn")).toBe(false);
  });
});
