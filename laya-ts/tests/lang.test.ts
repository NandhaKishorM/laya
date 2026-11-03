// laya-ts/tests/lang.test.ts
import { describe, expect, it } from "vitest";
import { analyse, detectScript, guessLatinLanguage, isEnglish, stateText } from "../src/lang.js";
import { Router } from "../src/router.js";
describe("lang", () => {
  it("detects devanagari as non-latin", () => {
    expect(detectScript("मुझसे दो बार शुल्क लिया गया")).toBe("devanagari");
  });
  // Python's state_text slices its 4000-character budget by code points; this port sliced by
  // UTF-16 units, so a state whose 4000th unit was the high half of an emoji returned text
  // ending in an orphaned surrogate -- a malformed string Python never produces, and the same
  // lone-surrogate shape laya-serve already refuses as a request (#454, #814). The email port
  // carries cutCodePoints for the same reason ("a budget cut an emoji in half").
  // Expectations from Python's laya.lang.state_text on the same input.
  it("stateText cuts its budget at code points, never splitting a surrogate pair", () => {
    const state = "A".repeat(3999) + "\uD83D\uDE00" + "tail text that will be cut";
    const text = stateText(state);
    expect([...text].length).toBe(4000);
    expect(text.codePointAt(text.length - 2)).toBe(0x1f600); // the whole emoji survived
    expect(text.endsWith("\uD83D\uDE00")).toBe(true);      // no orphaned high surrogate at the cut
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
  // Python divides the diacritic count by len(lowered) -- code points. This port divided by
  // `lowered.length`, which counts UTF-16 units, so every astral character (an emoji, an Old
  // Italic letter) counted as two and understated the rate. The first case sits at the
  // NON_EN_DIACRITIC_RATE threshold: Python's rate is 1/49 = 0.0204 >= 0.02 (non-English, off the
  // English checkpoint), the UTF-16 rate is 1/51 = 0.0196 < 0.02, and `analyse` answered
  // `isEnglish: true` here while Python answered `false` -- two emoji changed the routing verdict.
  // The other two show the rate itself. Expectations from Python's laya.lang.latin_profile on the
  // same inputs.
  it("counts code points, not UTF-16 units, in diacriticRate", () => {
    const flip = analyse("é\uD83D\uDE00\uD83D\uDE00" + "-".repeat(46));
    expect(flip.diacriticRate).toBeCloseTo(0.0204, 6);
    expect(flip.isEnglish).toBe(false);
    const heavy = analyse("créé résumé résumé résumé " + "\uD83D\uDE00".repeat(20));
    expect(heavy.diacriticRate).toBeCloseTo(0.1739, 4);
    expect(heavy.isEnglish).toBe(false);
    const spanish = analyse("día señor niño " + "\uD83D\uDE00".repeat(15));
    expect(spanish.diacriticRate).toBeCloseTo(0.1, 6);
    expect(spanish.isEnglish).toBe(false);
  });
  // Vietnamese: no stopword list of its own, so NON_EN_DIACRITICS is the only signal that keeps
  // it off the English checkpoint -- and the tone-marked letters are precomposed codepoints the
  // set did not hold, so everyday sentences carried a diacritic rate of 0.0 and read as English.
  // Expectations generated with Python's laya.lang.analyse on the same inputs.
  it("counts Vietnamese tone-marked letters as non-English", () => {
    const a = analyse("Cảm ơn bạn nhiều nhẻ");
    expect(a.diacriticRate).toBeCloseTo(0.25, 5);
    expect(a.isEnglish).toBe(false);
    expect(a.language).toBe(null);
    const b = analyse("Mình muốn hỏi về đơn hàng đã đặt");
    expect(b.diacriticRate).toBeCloseTo(0.3438, 4);
    expect(b.isEnglish).toBe(false);
  });
  it("keeps English with Vietnamese loanwords and names English", () => {
    expect(isEnglish("The pho was great and the banh mi was the best in town")).toBe(true);
    expect(isEnglish("Please help Nguyễn with the invoice for the last order")).toBe(true);
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
  it("names English made of question words and contractions", () => {
    for (const text of ["Why hasn't my card been delivered?", "How long does a card delivery take?",
      "Where should my refund show up?"]) {
      expect(guessLatinLanguage(text)).toBe("en");
      expect(analyse(text).languageUndecided).toBe(false);
    }
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

  // mixed-segment detection (port of Python's _non_english_segment)
  // A foreign line embedded in an English ticket must flip isEnglish and record the segment.
  const TRACE = "I checked the refund status and here is what I found for the customer.\n" +
    "The charge cleared last Tuesday and the bank confirmed it went through.\n" +
    "Quero o meu dinheiro de volta agora mesmo porque já esperei demais\n" +
    "Please escalate to the billing manager if the refund has not been issued by Friday.";

  it("mixed-segment: Portuguese line in English ticket is not english", () => {
    const a = analyse(TRACE);
    expect(a.isEnglish).toBe(false);
    expect(a.language).toBe("pt");
    expect(a.mixedSegment).toBe(
      "Quero o meu dinheiro de volta agora mesmo porque já esperei demais",
    );
  });

  it("mixed-segment: German error payload in English ticket", () => {
    const state = "The integration test failed with this error on the staging server " +
      "and I am not sure whether it is a data issue or a code regression.\n" +
      "Fehler: Die Verbindung zum Server wurde unterbrochen, bitte versuchen Sie es spaeter noch einmal";
    const a = analyse(state);
    expect(a.isEnglish).toBe(false);
    expect(a.mixedSegment).toContain("Verbindung");
  });

  it("mixed-segment: Spanish error in structured state", () => {
    const state = {
      subject: "Payment failed for a customer in Madrid",
      description: "The customer tried three times with the same card and each attempt was declined by " +
        "the gateway, so we would like to know whether the problem is on our side or with the bank.",
      error: {
        code: "card_declined",
        message: "La tarjeta fue rechazada por el banco emisor, contacte con su banco",
      },
    };
    const a = analyse(state);
    expect(a.isEnglish).toBe(false);
    expect(a.mixedSegment).toContain("tarjeta");
  });

  it("mixed-segment: all-caps Portuguese line", () => {
    const state =
      "This is the fourth email I have sent about the same order and nobody has answered any of them.\n" +
      "The customer wrote this in the chat and then closed the window:\n" +
      "QUERO MEU DINHEIRO DE VOLTA AGORA\n" +
      "Could someone from the billing team look at order 5512 today?";
    const a = analyse(state);
    expect(a.isEnglish).toBe(false);
    expect(a.mixedSegment).toBe("QUERO MEU DINHEIRO DE VOLTA AGORA");
  });

  // English stays English: multi-line, short foreign sign-off, code lines
  it.each([
    ["multi-line english", "Hi team,\nThe export failed again last night.\nCan you check the logs?\nThanks"],
    ["short portuguese sign-off", "Please resend the invoice for March, the amount is wrong.\nAtenciosamente, Joao"],
    ["os.path", "The build broke after the refactor.\nREPO = os.path.dirname(os.path.dirname(__file__))\n" +
      "Please take a look at the import paths when you can."],
    ["round(el)", "The latency script crashes on large runs.\nmix[key] = {\"total_s\": round(el, 2)}\n" +
      "Can you check why the stream is empty?"],
    ["same word twice (Nav/Com)", "I'm looking for good deals on the following (used or new):\n" +
      "Aviation Headsets (with mic).\n" +
      "Handheld Nav/Com tranciever (may consider COM only).\nPortable GPS or Loran Navigator."],
    ["team codes", "Round two predictions for the pool, as promised.\nQUE  vs MON:  MON  in 7.\n" +
      "PIT  vs NYI:  PIT  in 5."],
    ["slash compound", "I need a converter for these image formats.\n" +
      "DOS, OS/2 or platform independent programs if possible.\nThanks in advance."],
    ["backslash path", "My modem stopped answering after the upgrade.\nC:\\DOS\\mode COM1:9600,n,8,1,p\n" +
      "Is that the right line for a 9600 baud connection?"],
  ])("mixed-segment english stays english: %s", (_label, state) => {
    expect(analyse(state).isEnglish).toBe(true);
    expect(analyse(state).mixedSegment).toBeNull();
  });

  // The segment check reads at most 4000 chars, so a foreign line past that cap is not seen.
  it("mixed-segment: segment check reads at most the cap", () => {
    const state = {
      log: "The export failed again last night for the whole region. ".repeat(80),
      body: "Quero cancelar meu plano agora mesmo",
    };
    expect(analyse(state).mixedSegment).toBeNull();
  });

  it.each([
    "Please send me the café menu today please",
    "Could you email me your résumé before the meeting",
    "Send the invoice to José before Friday",
    "We visited Zürich last summer and loved it",
  ])("keeps English with one accented loanword English: %s", (text) => {
    expect(guessLatinLanguage(text)).toBe("en");
    expect(isEnglish(text)).toBe(true);
  });

  it.each<[string, string | null]>([
    ["Grüße aus Köln, wir melden uns wegen der Rechnung", "de"],
    ["sluk lyset i soveværelset", null],
    ["kan jeg få en refundering for det dobbelte beløb", null],
    ["stäng av ljuset i sovrummet", null],
    ["jag vill ha en återbetalning för den dubbla avgiften", "sv"],
    ["The naïve façade needs a fresh coat of paint", null],
  ])("does not rescue accented non-English text: %s", (text, language) => {
    expect(guessLatinLanguage(text)).toBe(language);
    expect(isEnglish(text)).toBe(false);
  });

  // A foreign function word that is also an ordinary English word (`come` it, `son` es, `do` pt,
  // `care` ro, `todo`/`im`/`per`/`plus`) named that language when it merely appeared twice, so
  // plain English routed to the multilingual checkpoint. Such a word now counts once however often
  // it repeats. Ported from tests/test_lang_guess.py; expectations from Python's laya.lang.analyse.
  it.each([
    "Come one, come all",
    "My son, your son",
    "Do more, do less",
    "Care more, care less",
    "Add a todo, then another todo item",
    "im not able to log in, im stuck",
    "add 45 to 87 plus 54 plus 43 plus 22",
  ])("plain english with a repeated collision word stays english: %s", (text) => {
    expect(analyse(text).isEnglish).toBe(true);
  });

  // The dedupe is limited to those words: `der`, `des`, `di`, `sa` are nobody's English and still
  // count every occurrence, Dutch `van` is deliberately left off the list, and genuine requests
  // with several different function words are still foreign.
  it.each([
    "reduzieren der helligkeit der lichter",
    "enumerar todos los horarios de los tren a nueva york",
    "jouer des chansons des beatles",
    "numero di telefono di giacomo",
    "pede um pacote de massa chinesa faz um pedido takeaway",
    "ar trebui sa port o pelerina de ploaie inainte sa ies afara",
    "herinner me eraan dat ik dat liedje leuk vind",
    "hey olly ik hou van muziek van frans bauer",
    "Hola, necesito cancelar mi pedido por favor ahora mismo gracias",
    "Eu quero cancelar o meu pedido por favor agora mesmo obrigado",
    "Vorrei annullare il mio ordine per favore adesso grazie mille",
  ])("a repeated non-collision function word is still foreign: %s", (text) => {
    expect(analyse(text).isEnglish).toBe(false);
  });
});

// `leafNonEnglish` caps each line at 4000 characters, the same way Python's `_leaf_non_english`
// does (#1449). Sampling the first 4000 characters of the whole leaf hid a foreign line that
// followed a longer English prefix, and it sent that state to the English checkpoint. A single
// line still stops at 4000: foreign text pasted onto the end of one long line is not read.
// Expectations below are Python's `laya.lang.analyse` on the same inputs.
describe("lang/leaf scan reads each line", () => {
  const ENGLISH_LINE = "The customer reports that the invoice was issued twice this quarter and asks for a refund. ";
  // 91 * 50 = 4550, past the per-line cap, so the following line is outside one whole-leaf sample.
  const ENGLISH_PREFIX = ENGLISH_LINE.repeat(50);
  const GERMAN = "Ich wurde doppelt belastet und möchte eine Rückerstattung";
  const GERMAN_ASCII = "Ich wurde doppelt belastet und moechte eine Rueckerstattung";

  it("finds a German line after an English prefix longer than 4000 characters", () => {
    const got = analyse({ note: `${ENGLISH_PREFIX}\n${GERMAN}` });
    expect(got.isEnglish).toBe(false);
    expect(got.language).toBe("de");
    expect(got.languageUndecided).toBe(false);
    expect(got.mixedSegment).toBeNull();
  });

  it("finds ASCII German after the same prefix", () => {
    const got = analyse({ note: `${ENGLISH_PREFIX}\n${GERMAN_ASCII}` });
    expect(got.isEnglish).toBe(false);
    expect(got.language).toBe("de");
    expect(got.mixedSegment).toBeNull();
  });

  it("still names German when that line comes before the English prefix", () => {
    const got = analyse({ note: `${GERMAN}\n${ENGLISH_PREFIX}` });
    expect(got.isEnglish).toBe(false);
    expect(got.language).toBe("de");
    expect(got.mixedSegment).toBe(GERMAN);
  });

  it("does not read past 4000 characters of one line", () => {
    // No newline, so the German tail sits past the cap and the line is English.
    const sameLine = analyse({ note: ENGLISH_PREFIX + GERMAN });
    expect([...ENGLISH_PREFIX].length).toBeGreaterThan(4000);
    expect(sameLine.isEnglish).toBe(true);
    expect(sameLine.language).toBe("en");
    expect(sameLine.mixedSegment).toBeNull();

    // The line is longer than the cap, and its first 4000 characters are still German.
    // An English sibling fills the segment budget, so only the per-line leaf read can see it.
    const longGerman = `${GERMAN}. `.repeat(80);
    expect([...longGerman].length).toBeGreaterThan(4000);
    const ownLine = analyse({ note: "Please refund this. ".repeat(200), body: longGerman });
    expect(ownLine.isEnglish).toBe(false);
    expect(ownLine.language).toBe("de");
    expect(ownLine.mixedSegment).toBeNull();
  });

  it("skips empty lines and still reads the foreign line after them", () => {
    const afterBlanks = analyse({ note: `${ENGLISH_PREFIX}\n\n\n${GERMAN}` });
    expect(afterBlanks.isEnglish).toBe(false);
    expect(afterBlanks.language).toBe("de");

    const afterWhitespace = analyse({ note: `${ENGLISH_PREFIX}\n   \n\t\n${GERMAN}` });
    expect(afterWhitespace.isEnglish).toBe(false);
    expect(afterWhitespace.language).toBe("de");

    const onlyEmpty = analyse({
      note: "Please refund the duplicate charge on invoice 4411 today.",
      extra: "\n\n\n",
    });
    expect(onlyEmpty.isEnglish).toBe(true);
    expect(onlyEmpty.language).toBe("en");

    const onlySpaces = analyse({
      note: "Please refund the duplicate charge on invoice 4411 today.",
      extra: " ".repeat(5000),
    });
    expect(onlySpaces.isEnglish).toBe(true);
    expect(onlySpaces.language).toBe("en");
  });

  it("keeps the most letter-rich qualifying line of one leaf", () => {
    // Python's inner scan does not return on the first hit. "status" is six characters and
    // is skipped; the short German line qualifies; the repeated Portuguese line has more letters.
    const portuguese = "Quero o meu dinheiro de volta agora mesmo porque já esperei demais";
    const got = analyse({
      note: "The customer was billed twice and wants a refund. ".repeat(140),
      long: `status\nMein Konto wurde zweimal belastet\n${portuguese.repeat(3)}`,
    });
    expect(got.isEnglish).toBe(false);
    expect(got.language).toBe("pt");
    expect(got.mixedSegment).toBeNull();
  });
});

// All-caps handling, ported from `laya/lang.py:605-660, 750-800` (#1315's follow-up). An acronym
// block is not a foreign language, so it must not vote in the whole-state verdict nor name a
// foreign segment on its own line; a customer shouting in Portuguese is still Portuguese.
//
// The states and expectations below are the Python reference's own, recorded in
// `laya-rust/tests/golden/routing/lang_probe.json` (generated by `laya/lang.py`). Before the port
// nine of those recorded cases diverged; they are these.
describe("lang/all-caps routing", () => {
  const recorded: [string, string, unknown, string | null, boolean][] = [
    // label, comment, state, expected language, expected isEnglish
    ["shouted acronyms", "an acronym line alone is not French", "MON DES EST LA", null, true],
    ["acronyms under one English line", "", "Please look at this\nMON DES EST LA", "en", true],
    ["acronyms under two English lines", "", "Please look at this\nand send me the details.\nMON DES EST LA", "en", true],
    ["shouted address", "US cities, not Spanish", "STORES LOS ANGELES LAS VEGAS EL PASO CLOSED", null, true],
    ["shouted German, umlaut", "diacritics are evidence, so it stays non-English", "WIE SPÄT IST ES IN KÖLN", null, false],
    ["shouted German, ASCII", "no diacritics and no long stopword: the documented loss", "WIE SPAET IST ES IN KOELN", null, true],
    ["emphasis capitals in Portuguese", "a word-length run is not an acronym and is not blanked", "quero cancelar o pedido MON DES EST LA por favor", "pt", false],
    ["acronym segment beside prose", "", "Please review the error and send me the details.\nMON DES EST LA", "en", true],
  ];
  it.each(recorded)("%s: %s", (label, _comment, state, language, isEnglish) => {
    const got = analyse(state);
    expect(got.language, label).toBe(language);
    expect(got.isEnglish, label).toBe(isEnglish);
  });

  it("routes ordinary English support text carrying an acronym line to the English checkpoint", () => {
    // The reported shape: real mail with a technical block in capitals. Python routes it english;
    // this side named the block Portuguese and took the multilingual checkpoint.
    const state = "Hello,\nNAV COM OS 2 DESKTOP SUPPORT\nI cannot connect to the VPN";
    expect(analyse(state).isEnglish).toBe(true);
    expect(new Router().route(state, {}).model).toBe("english");
  });

  it("keeps a shouted foreign line foreign when it carries diacritics", () => {
    // `shoutedEvidence`'s other branch: non-English diacritics count in place of a long stopword,
    // so the veto must not swallow a customer writing in capitals in their own language.
    const state = "WIE SPÄT IST ES IN KÖLN";
    expect(analyse(state).isEnglish).toBe(false);
    expect(new Router().route(state, {}).model).toBe("multilingual");
  });

  it("does not blank a word-length run out of its own sentence", () => {
    // Blanking every all-caps run deleted emphasis capitals and lost verdicts (`sag mir das
    // HEUTIGE DATUM` routed english). Only runs short enough to be acronyms are blanked.
    expect(analyse("quero cancelar o pedido MON DES EST LA por favor").language).toBe("pt");
  });
});
