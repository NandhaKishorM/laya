// laya-ts/tests/lang.test.ts
import { describe, expect, it } from "vitest";
import { analyse, detectScript, guessLatinLanguage, isEnglish } from "../src/lang.js";
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
  // The Azerbaijani stopword list above has always been in the port, but `ə` was missing from
  // NON_EN_DIACRITICS, so the one letter that identifies the language carried no weight. English
  // text with a single schwa read as English here and as undecided in Python. Expectations from
  // Python's laya.lang.analyse on the same inputs.
  it("counts the Azerbaijani schwa as a non-English letter", () => {
    const a = analyse("və the quick brown");
    expect(a.diacriticRate).toBeCloseTo(0.0556, 3);
    expect(a.isEnglish).toBe(false);
    expect(a.language).toBe(null);
  });
  it("still routes plain english without a schwa to english", () => {
    const a = analyse("the quick brown fox");
    expect(a.diacriticRate).toBe(0);
    expect(a.isEnglish).toBe(true);
    expect(a.language).toBe("en");
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
  // Plain-ASCII German: no umlaut, so the diacritic rate is 0 and the function words are the only
  // evidence. Python names these `de` (laya/lang.py, #130); with the 23-word list they read as
  // English here and the Router sent them to the English checkpoint.
  it("names German that carries no diacritics", () => {
    for (const text of [
      "wie lautet die temperatur in fulda in hessen",
      "Mein Konto wurde zweimal belastet, bitte erstatten Sie",
      "Ich brauche eine Rechnung fuer meine letzte Bestellung",
      "Kann ich meine Bestellung noch heute stornieren",
      "Bitte senden Sie mir eine neue Kreditkartenabrechnung",
    ]) {
      const a = analyse(text);
      expect(a.language, text).toBe("de");
      expect(a.isEnglish, text).toBe(false);
      expect(guessLatinLanguage(text), text).toBe("de");
    }
  });
  // `in` and `was` count for German as well as English, and `im` and `den` are English tokens too.
  // Pin the English states that flip if any of them becomes German-only.
  it("keeps English that shares words with the German list", () => {
    expect(analyse("turn off smart lamp in den").isEnglish).toBe(true);
    expect(analyse("What was the reason for the delay").language).toBe("en");
    expect(analyse("Where can I find my invoice in the app").isEnglish).toBe(true);
  });
  // `es` and `du` are German function words, but Spanish and French claim them, and a word two
  // lists share names neither language. They stay out of the German list.
  it("does not pull es or du into German", () => {
    expect(analyse("que hora es en australia").language).toBe("es");
    expect(analyse("baisse le volume du haut-parleur").language).toBe("fr");
  });
  it.each<[string, string | null, boolean]>([
    ["Preciso do contrato confidencial assinado até sexta.", "pt", false],
    ["Gătește-mi o rețetă de sarmale de post pentru mâine.", "ro", false],
    ["Müşteriden iki kez ücret alındı ve para iadesi istiyor", null, false],
    ["Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", null, false],
  ])("counts accented words whole, like Python: %s", (text, language, english) => {
    const a = analyse(text);
    expect(a.language).toBe(language);
    expect(a.isEnglish).toBe(english);
    expect(a.languageUndecided).toBe(language === null);
  });

  it("does not let English sibling fields hide a German value", () => {
    const sentence = "Mein Konto wurde zweimal belastet";
    const asString = analyse(sentence);
    expect(asString.language).toBe("de");
    expect(asString.isEnglish).toBe(false);
    expect(analyse({ message: sentence }).language).toBe(asString.language);
    const buried = analyse({
      agent_notes: "Please check the shipping status and refund the customer if the charge was duplicated. The order was late and we have not heard back.",
      message: sentence,
    });
    expect(buried.isEnglish).toBe(false);
    expect(buried.language).toBe("de");
  });
});
