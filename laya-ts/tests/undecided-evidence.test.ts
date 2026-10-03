import { describe, expect, it } from "vitest";
import { Router, guessLatinLanguage } from "../src/index.js";
import { FOREIGN_WORDS } from "../src/foreign-words.js";

// Same cases as tests/test_router.py (#54): from four words up, undecided Latin text leaves the default
// only with evidence of another language. MASSIVE test utterances unless noted.
describe("undecided Latin text with evidence of another language", () => {
  const router = new Router();

  it.each([
    "streiche alle meine geplanten termine",          // de, `alle` and `meine`
    "me gustaría escuchar algunos buenos chistes divertidos", // es, one `í`
    "tolong mainkan lagu dari bruno mars",            // id, `tolong` and `dari`
    "batalkan alarm saya pukul tujuh pagi",           // id, `saya`
    "olly beritahu saya satu jenaka",                 // ms, `saya`
    "olly ceritake aku guyonan",                      // jv, `aku`
    "tafadhali washa plagi mahiri",                   // sw, `tafadhali`
    "huwag mo akong gisingin bukas",                  // tl, `bukas`
    "ken jy enige grappe",                            // af, `jy`
    "paid a siarad heddiw",                           // cy, `heddiw`
    "schakel de lampen uit",                          // nl, `uit`
    "vertel me wat er gebeurt op instagram",          // nl, `wat` and `op`
  ])("routes to multilingual: %s", (text) => {
    expect(guessLatinLanguage(text)).toBeNull();
    expect(router.route(text, {}).model).toBe("multilingual");
  });

  it("names the evidence in the reason", () => {
    expect(router.route("streiche alle meine geplanten termine", {}).reason)
      .toContain("function words of other languages");
    expect(router.route("me gustaría escuchar algunos buenos chistes divertidos", {}).reason)
      .toContain("a non-English letter");
  });

  it.each([
    "how do my health benefits work",                 // `do`, English texts checked on #286
    "las vegas weather today",
    "verificar qualquer email da amazon",             // pt, `da` alone
    "olly give me some dim light",                    // `dim` (cy), en-US
    "kung fu panda three",                            // `kung` (tl), en-US
    "tell me my current savings account's interest rate",
    "turn off lobby light",
    "wecke mich auf",                                 // under four words
  ])("follows the default without it: %s", (text) => {
    expect(router.route(text, {}).model).toBe("english");
    expect(new Router({ default: "multilingual" }).route(text, {}).model).toBe("multilingual");
  });

  it("keeps the generated words", () => {
    expect(FOREIGN_WORDS.has("saya")).toBe(true);
    expect(FOREIGN_WORDS.has("do")).toBe(false);
  });
});
