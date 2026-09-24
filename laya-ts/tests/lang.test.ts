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
  // Every expectation below was generated with Python's laya.lang.analyse on the same input.
  it("names romanized Bangla instead of falling back to english", () => {
    const a = analyse("ami ekta ticket khulsi, kalke theke payment hocche na, ekhon ki korte parbo");
    expect(a.script).toBe("latin");
    expect(a.language).toBe("bn");
    expect(a.isEnglish).toBe(false);
  });
  it("names Azerbaijani Latin text", () => {
    const a = analyse("ödənişim iki dəfə tutulub, amma heç bir təsdiq almadım, nə etməliyəm");
    expect(a.language).toBe("az");
    expect(a.isEnglish).toBe(false);
  });
  it("a CJK sentence inside an English ticket is not english", () => {
    const a = analyse("please check the attached logs 請重啟服務器然後再試一次 and tell me what failed");
    expect(a.script).toBe("han");
    expect(a.isEnglish).toBe(false);
  });
  it("a tiny non-latin note does not flip an english ticket", () => {
    const a = analyse("please refund my order, see note 重啟 attached to the ticket");
    expect(a.script).toBe("latin");
    expect(a.isEnglish).toBe(true);
  });
  it("counts IPA extensions as latin, like Python", () => {
    expect(detectScript("ɑ ɒ ɛ ɔ ɪ ʊ æ ʃ θ ð")).toBe("latin");
  });
});
