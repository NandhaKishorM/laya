// laya-ts/tests/state-script-fields.test.ts
import { describe, expect, it } from "vitest";
import { analyse, detectScript, scriptProfile, stateText } from "../src/lang.js";

// A long English note outnumbers a short message in another script, so the whole-state read is
// Latin and names no language, and the per-value pass is what flips `isEnglish`. Every expectation
// below was generated with Python's laya.lang.analyse on the same state.
const NOTE = "Please read the attached invoice carefully and let me know if you need anything else. ".repeat(20);

describe("analyse: a per-value override keeps the state's own script fields", () => {
  it.each([
    ["Мне нужен счет на оплату.", "cyrillic", 0.0141],
    ["我需要一张发票，请尽快处理，谢谢。", "han", 0.0099],
    ["χρειάζομαι ένα τιμολόγιο για την παραγγελία", "greek", 0.0264],
    ["أحتاج إلى فاتورة لهذا الطلب من فضلك", "arabic", 0.0203],
  ])("reports the state as Latin, not as the %s leaf", (msg, leafScript, share) => {
    const state = { note: NOTE, msg };
    const r = analyse(state);
    // The message still decides the route.
    expect(r.isEnglish).toBe(false);
    expect(r.language).toBeNull();
    // The script fields describe the state, which is overwhelmingly Latin. They used to describe
    // the leaf instead: `script: leafScript`, `scriptProfile: {[leafScript]: 1}`,
    // `nonLatinFraction: 1`.
    expect(r.script).toBe("latin");
    expect(r.scriptProfile[leafScript]).toBeCloseTo(share, 4);
    expect(r.nonLatinFraction).toBeCloseTo(share, 4);
  });

  it("keeps the result self-consistent with scriptProfile(stateText(state))", () => {
    const state = { note: NOTE, msg: "Мне нужен счет на оплату." };
    const r = analyse(state);
    expect(r.scriptProfile).toEqual(scriptProfile(stateText(state)));
    expect(r.script).toBe(detectScript(stateText(state)));
    expect(r.nonLatinFraction).toBeCloseTo(1 - (r.scriptProfile["latin"] ?? 0), 4);
  });

  it("leaves a state that is non-Latin as a whole alone", () => {
    // Here the whole-state read already names Cyrillic, so the override never runs and the fields
    // were correct before this change.
    const r = analyse({ msg: "Мне нужен счет на оплату.", note: "Спасибо" });
    expect(r.script).toBe("cyrillic");
    expect(r.isEnglish).toBe(false);
  });
});

describe("analyse: the reported rates round the way Python's round() does", () => {
  // 1 non-English letter in 32 characters is 0.03125 exactly, a tie at four decimals. Python's
  // round() takes the even digit (0.0312); Math.round took the larger one (0.0313).
  it("rounds a diacritic rate of 1/32 down", () => {
    expect(analyse("é" + "a".repeat(31)).diacriticRate).toBe(0.0312);
  });

  it("rounds a non-Latin fraction of 1/32 down", () => {
    expect(analyse("你" + "a".repeat(31)).nonLatinFraction).toBe(0.0312);
  });

  // 1 in 4000 reads as a tie in decimal but is not one in binary: the double is
  // 0.0002500000000000000052, so Python rounds up. Scaling by 10000 lands on 2.5 exactly and would
  // round this down, which is why the digits come from the double's own expansion.
  it("rounds a rate of 1/4000 up, as the double is above the halfway point", () => {
    expect(analyse("é" + "a".repeat(3999)).diacriticRate).toBe(0.0003);
  });

  it("still rounds away from a tie the same way as before", () => {
    // 3 diacritics in 32 characters is 0.09375; the digit before the tie is odd, so both runtimes
    // round up. This pins the branch that did not change.
    expect(analyse("ééé" + "a".repeat(29)).diacriticRate).toBe(0.0938);
  });
});
