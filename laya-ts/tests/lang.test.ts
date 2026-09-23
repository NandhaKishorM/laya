// laya-ts/tests/lang.test.ts
import { describe, expect, it } from "vitest";
import { analyse, detectScript, isEnglish } from "../src/lang.js";
describe("lang", () => {
  it("detects devanagari as non-latin", () => {
    expect(detectScript("मुझसे दो बार शुल्क लिया गया")).toBe("devanagari");
  });
  it("routes english latin to english", () => {
    expect(analyse("Please refund the duplicate charge").isEnglish).toBe(true);
  });
  it("routes german latin to non-english", () => {
    expect(isEnglish("Der Kunde wurde zweimal belastet")).toBe(false);
  });
  it("unknown (no letters) is english + undecided", () => {
    expect(analyse("123 !!!").script).toBe("unknown");
  });
});
