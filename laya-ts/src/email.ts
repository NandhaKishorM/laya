// Python's `re` classes and its `len()`/slicing are Unicode-wide and count code points; JS's `\s`,
// `\d`, `\w` and `\b` are ASCII-ish and `.length`/`.slice` count UTF-16 units. The markers below are
// compiled from the three classes and the two helpers at the bottom of this block, because each
// difference is a case the reference records in `laya-java/fixtures/email.json` and this port got
// wrong: a `\x1c` before `On ... escreveu:` was not indentation, an Arabic-Indic or Devanagari date
// was not a date, a cuneiform sign was not a word character (so a sender's own sentence read as a
// disclaimer and was deleted whole), and a budget cut an emoji in half.
//
// The space class is the measured one: it is exactly the set `re.match(r"\s", ch)` accepts on
// CPython, which is also the set `str.strip()` removes. JS's `\s` omits U+001C-U+001F and U+0085 and
// adds U+FEFF, so both the class and `trimPy` below stand in for it.
const SP_IN =
  "\\u0009-\\u000D\\u001C-\\u001F\\u0020\\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029" +
  "\\u202F\\u205F\\u3000";
const SP = `[${SP_IN}]`; // Python's \s
const NOT_SP = `[^${SP_IN}]`; // Python's \S
// Python's \w is `str.isalnum()` plus the underscore: the letter and number categories and nothing
// else -- measured over the whole repertoire, marks (Mn/Mc/Me) and the join controls are NOT word
// characters, and every code point CPython accepts is in `[\p{L}\p{N}_]`. The class is wider by the
// 4,657 code points JS's newer Unicode tables have assigned since CPython's, none of which is a
// letter or digit Python knows.
const WORD_IN = "\\p{L}\\p{N}_"; // the class body Python's \w accepts
const WORD = `[${WORD_IN}]`; // Python's \w
const DIGIT = "\\p{Nd}"; // Python's \d is the Nd category alone
const B_L = `(?<!${WORD})`; // Python's \b before a word character
const B_R = `(?!${WORD})`; // Python's \b after a word character

const LEADING_SP = new RegExp(`^${SP}+`, "u");
const TRAILING_SP = new RegExp(`${SP}+$`, "u");

/** Python's `str.strip()`, over the space set `re`'s `\s` uses. */
function trimPy(text: string): string {
  return text.replace(LEADING_SP, "").replace(TRAILING_SP, "");
}

/** Python's `len()` on a string: code points, not the UTF-16 units `.length` counts. */
function width(text: string): number {
  let count = 0;
  for (const _ch of text) count++;
  return count;
}

/** Python's `text[:count]`, cutting at code points so a surrogate pair is never split in half. */
function cutCodePoints(text: string, count: number): string {
  const chars = Array.from(text);
  const end = count < 0 ? Math.max(0, chars.length + count) : Math.min(count, chars.length);
  return chars.slice(0, end).join("");
}

const QUOTE_HEADERS: RegExp[] = [
  new RegExp(`^${SP}*On .{0,300}wrote:${SP}*$`, "iu"),
  // "Em resposta ao que você escreveu:" is body text; a client's attribution always carries a date
  new RegExp(`^${SP}*Em (?=.*${DIGIT}).{0,300}escreveu:${SP}*$`, "iu"),
  new RegExp(`^${SP}*El (?=.*${DIGIT}).{0,300}escribi[óo]:${SP}*$`, "iu"),
  // Gmail's French attribution carries a date like the others ("Le lun. ... a écrit :"), and
  // French typography puts a space before the colon, so the verb allows one
  new RegExp(`^${SP}*Le (?=.*${DIGIT}).{0,300}a [eé]crit${SP}*:${SP}*$`, "iu"),
  new RegExp(`^${SP}*-{2,}${SP}*(Original|Forwarded) Message${SP}*-{2,}`, "iu"),
  new RegExp(
    `^${SP}*-{2,}${SP}*(Mensagem (original|encaminhada)|Mensaje (original|reenviado)|` +
      "Message d'origine)" +
    `${SP}*-{2,}`,
    "iu",
  ),
  new RegExp(`^${SP}*_{8,}${SP}*$`, "u"),
  // `From:` opens ordinary prose too ("From: my side the integration works, but please
  // refund..."), and a reply header always carries the sender, so the header is only recognised
  // when an address follows -- the same rule as `De:` below. A bare `From: Name` header is
  // caught by HEADER_FROM_NAME/HEADER_NEXT instead, which need the header's own `Sent:`/`Date:`
  // line to tell it apart from a sentence.
  new RegExp(`^${SP}*From:${SP}.*[@<]`, "iu"),
  // `De:` also opens ordinary Portuguese/Spanish lines ("De: 10/09 a 15/09"), so the Outlook
  // header is only recognised when it carries an address. French Outlook writes `De :` with a
  // space, so the marker allows one; the address rule is unchanged
  new RegExp(`^${SP}*De${SP}*:${SP}.*[@<]`, "iu"),
];
const ATTRIBUTION_TAIL = new RegExp(
  `^.{0,120}${NOT_SP}@${NOT_SP}+${SP}+(wrote|escreveu|escribi[óo]|a [eé]crit)${SP}*:${SP}*$`,
  "iu",
);
const ATTRIBUTION_HEAD = new RegExp(`^${SP}*(On|Em|El|Le) (?=.*${DIGIT})`, "iu");
// Exchange often leaves the address out of Outlook's reply header ("De: Maria Souza"), so a bare
// `De:` only cuts when the header's own `Enviado:` line, or a dated `Data:`/`Fecha:` line, follows
// it. `Para:` is not enough: "De: 10/09 / Para: 15/09" is how a leave request reads.
//
// The same is true of a bare English `From: Maria Souza`, which is why the marker above needs
// this rule: the English client lines are the translations of the two `De:` neighbours. A line
// that only looks like prose still has to be told apart from a header by its neighbours, so the
// English pair is "From: <name>" followed by "Sent:"/"Date:". French Outlook writes the same
// header as `De : Marie Dupont` with `Envoyé :` underneath, so both allow the spaced colon.
const HEADER_FROM_NAME = new RegExp(`^${SP}*(De|From)${SP}*:${SP}+${NOT_SP}`, "iu");
const HEADER_NEXT = new RegExp(
  `^${SP}*(Enviad[oa]( em| el)?:${SP}|Envoy[ée]( le)?${SP}*:${SP}|Sent:${SP}|` +
    `(Data|Fecha|Date):${SP}.*${DIGIT}{4})`,
  "iu",
);
// A closing line is the closing word plus punctuation and at most a name. Anything else on
// the line is a sentence, and the case of the next word is what separates the two: a name is
// capitalised, "for" in "Thanks for the quick reply." is not. JS regexes have no scoped
// case-insensitive groups like Python's (?i:...), so the closing words are matched
// case-insensitively here and the name is checked case-sensitively by SIGNOFF_TAIL.
const SIGNOFF_HEAD = new RegExp(
  `^${SP}*(?:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)` +
    `(?:${SP}+(?:and|&)${SP}+regards|${SP}+(?:regards|wishes|again|in advance|a lot|so much|very much))?`,
  "iu",
);
// The name's first letter must not be lowercase, so the check is per character, not per script:
// \p{Lu}/\p{Lt}/\p{Lo} accepts a name capitalised in any script (Regards, Łukasz) or written in
// a script without case (山田). Python asks the same question per token (`_is_english_signoff`).
const SIGNOFF_TAIL = new RegExp(
  `^[${SP_IN},;:!.]*(?:[\\p{Lu}\\p{Lt}\\p{Lo}][\\p{L}\\p{M}\\p{N}_'-]*[${SP_IN},.]*){0,3}$`,
  "u",
);
function isEnglishSignoff(line: string): boolean {
  const m = SIGNOFF_HEAD.exec(line);
  return m !== null && SIGNOFF_TAIL.test(line.slice(m[0].length));
}
const SIGNATURE_MARKERS: Array<(line: string) => boolean> = [
  (l) => new RegExp(`^${SP}*--${SP}*$`, "u").test(l),
  isEnglishSignoff,
  (l) =>
    new RegExp(
      `^${SP}*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]` +
        "|(muito )?obrigad[oa]s?( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações" +
        "|(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?" +
        "|(bien )?cordialement|salutations( distinguées)?|bien à vous|merci( d'avance)?|bonne journée)" +
        `[${SP_IN},!.]*$`,
      "iu",
    ).test(l),
];
const DEVICE =
  "iphone|ipad|android|ios|mobile|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|" +
  "outlook|yahoo|mail|e-?mail|gmail|windows";
const DEVICE_FOOTER = new RegExp(
  `^${SP}*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?|` +
    "envoy[ée] (depuis|de)( mon| ma| mes)?)" +
    ` (${DEVICE})( (${DEVICE}|para|for|no|na|${DIGIT}+|phone|device|pro|max|mini|plus|` +
    `using [a-z][${WORD_IN}.+-]*))*` +
    `|(obter o|get) outlook (para|for) (ios|android))[${SP_IN}.!]*$`,
  "iu",
);
const DISCLAIMER = new RegExp(
  // English is tied to a disclaimer noun and a disclaimer tail, the way the Portuguese branches
  // below are. The bare word matched any sentence that merely mentioned it, so "Is this
  // confidential?" and "Confidential: I need a refund." were deleted whole and the model was
  // scored on an empty state. `[^.]` rather than `[^.\n]`: a footer wraps, so "are\nconfidential"
  // must still match.
  `(${B_L}(e-?mail|message|information|communication|transmission|contents?)${B_R}[^.]{0,60}` +
    `${B_L}confidential${B_R}[^.]{0,60}${B_L}(intended|solely|addressee|recipient|privileged|` +
    "disclos|unauthori[sz]ed)|" +
    `${B_L}confidential${B_R}[^.]{0,60}${B_L}(and (may|is) (also )?privileged)|` +
    "if you (have )?received this (e-?mail|message) in error|" +
    `${B_L}(esta|este) (mensagem|e-?mail|mensaje|correo)${B_R}[^.]{0,80}(confidencia|sigilos|privilegiad)|` +
    `${B_L}(uso exclusivo|exclusivamente|únicamente|unicamente)${B_R}[^.]{0,30}` +
    "(destinatári|destinatari|pessoa|persona|entidade|entidad)|" +
    `${B_L}(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)${B_R}[^.]{0,20} por (engano|erro)|` +
    `${B_L}(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)${B_R}[^.]{0,20} por error|` +
    `${B_L}antes de imprimir${B_R}[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|` +
    `${B_L}(meio|medio) ambiente${B_R}[^.]{0,30}antes de imprimir|` +
    // French is tied to "ce message/cet e-mail" rather than the bare word `confidentiel`, which a
    // sender's own request ("le contrat confidentiel") uses just as often
    `${B_L}(ce|cet|cette) (message|e-?mail|mail|courriel)${B_R}[^.]{0,80}(confidentiel|privil[eé]gi)|` +
    `${B_L}avez re[çc]u (ce|cet|cette) (message|e-?mail|mail)${B_R}[^.]{0,20} par erreur|` +
    `${B_L}(usage exclusif|exclusivement|uniquement)${B_R}[^.]{0,30}destinataire)`,
  "iu",
);
const SENTENCE = new RegExp(`(?<=[.!?])${SP}+`, "u");

function startsNewSentence(line: string): boolean {
  for (const ch of line) {
    if (/\p{L}/u.test(ch)) return ch === ch.toUpperCase() && ch !== ch.toLowerCase();
  }
  return false;
}

function splitFusedLines(sentence: string): string[] {
  if (!sentence.includes("\n")) return [sentence];
  const pieces: string[] = [];
  let buf = "";
  for (const line of sentence.split("\n").map(trimPy)) {
    if (!line) continue;
    if (buf && startsNewSentence(line)) {
      pieces.push(buf);
      buf = line;
    } else {
      buf = buf ? buf + " " + line : line;
    }
  }
  if (buf) pieces.push(buf);
  return pieces;
}

function stripDisclaimer(paragraph: string): string {
  if (!DISCLAIMER.test(paragraph)) return paragraph;
  const parts = paragraph.split(SENTENCE).map(trimPy).filter(Boolean);
  const pieces: string[] = [];
  for (const p of parts) {
    if (DISCLAIMER.test(p)) pieces.push(...splitFusedLines(p));
    else pieces.push(p);
  }
  return pieces.filter((p) => !DISCLAIMER.test(p)).join(" ");
}

export function cleanEmailBody(body: string, maxChars = 3000): string {
  let text = (body ?? "").replace(/\r\n?/g, "\n").replace(/\\n/g, "\n");
  // Bound regex work before the expensive patterns below (Python does the same): only
  // maxChars are ever returned, so nothing past 4x can survive cleaning.
  if (width(text) > maxChars * 4) text = cutCodePoints(text, maxChars * 4);
  const lines: string[] = [];
  const src = text.split("\n");
  for (let i = 0; i < src.length; i++) {
    const line = src[i];
    if (QUOTE_HEADERS.some((p) => p.test(line)) && lines.length) break;
    if (lines.length && HEADER_FROM_NAME.test(line) && i + 1 < src.length && HEADER_NEXT.test(src[i + 1])) break;
    if (ATTRIBUTION_TAIL.test(line) && lines.length) {
      if (ATTRIBUTION_HEAD.test(lines[lines.length - 1])) lines.pop();
      break;
    }
    if (line.replace(LEADING_SP, "").startsWith(">")) continue;
    lines.push(line.replace(TRAILING_SP, ""));
  }
  let cut = lines.length;
  const start = Math.max(1, Math.min(Math.floor(lines.length * 0.6), lines.length - 8));
  for (let i = start; i < lines.length; i++) {
    const n = width(trimPy(lines[i]));
    if (
      (n <= 40 && SIGNATURE_MARKERS.some((p) => p(lines[i]))) ||
      (n <= 60 && DEVICE_FOOTER.test(lines[i]))
    ) {
      cut = i;
      break;
    }
  }
  const kept = lines.slice(0, cut);
  const paragraphs = kept
    .join("\n")
    .split(new RegExp(`\\n${SP}*\\n`, "u"))
    .map(stripDisclaimer);
  const out = paragraphs
    .map(trimPy)
    .filter(Boolean)
    .join("\n\n")
    .replace(/[ \t]+/g, " ");
  return cutCodePoints(out, maxChars);
}

export function emailState(
  subject: string,
  body: string,
  sender?: string | null,
  clean = true,
  extra: Record<string, unknown> = {},
  // The budget cleanEmailBody cuts the body to, as Python's email_state(max_chars=) (#589).
  // Last, so existing positional calls keep their meaning. Ignored when clean is false.
  maxChars = 3000,
): Record<string, unknown> {
  const state: Record<string, unknown> = {
    subject: trimPy(subject ?? ""),
    body: clean ? cleanEmailBody(body ?? "", maxChars) : (body ?? ""),
  };
  if (sender) state["from"] = sender;
  for (const [k, v] of Object.entries(extra ?? {})) {
    if (v !== null && v !== undefined) state[k] = v;
  }
  return state;
}
