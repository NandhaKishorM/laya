const QUOTE_HEADERS: RegExp[] = [
  /^\s*On .{0,300}wrote:\s*$/i,
  /^\s*Em (?=.*\d).{0,300}escreveu:\s*$/i,
  /^\s*El (?=.*\d).{0,300}escribi[óo]:\s*$/i,
  /^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}/i,
  /^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado))\s*-{2,}/i,
  /^\s*_{8,}\s*$/,
  // `From:` opens ordinary prose too ("From: my side the integration works, but please
  // refund..."), and a reply header always carries the sender, so the header is only recognised
  // when an address follows -- the same rule as `De:` below. A bare `From: Name` header is
  // caught by HEADER_FROM_NAME/HEADER_NEXT instead, which need the header's own `Sent:`/`Date:`
  // line to tell it apart from a sentence.
  /^\s*From:\s.*[@<]/i,
  // `De:` also opens ordinary Portuguese/Spanish lines ("De: 10/09 a 15/09"), so the Outlook
  // header is only recognised when it carries an address
  /^\s*De:\s.*[@<]/i,
];
const ATTRIBUTION_TAIL = /^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]):\s*$/i;
const ATTRIBUTION_HEAD = /^\s*(On|Em|El) (?=.*\d)/i;
// Exchange often leaves the address out of Outlook's reply header ("De: Maria Souza"), so a bare
// `De:` only cuts when the header's own `Enviado:` line, or a dated `Data:`/`Fecha:` line, follows
// it. `Para:` is not enough: "De: 10/09 / Para: 15/09" is how a leave request reads.
//
// The same is true of a bare English `From: Maria Souza`, which is why the marker above needs
// this rule: the English client lines are the translations of the two `De:` neighbours. A line
// that only looks like prose still has to be told apart from a header by its neighbours, so the
// English pair is "From: <name>" followed by "Sent:"/"Date:".
const HEADER_FROM_NAME = /^\s*(De|From):\s+\S/i;
const HEADER_NEXT = /^\s*(Enviad[oa]( em| el)?:\s|Sent:\s|(Data|Fecha|Date):\s.*\d{4})/i;
// A closing line is the closing word plus punctuation and at most a name. Anything else on
// the line is a sentence, and the case of the next word is what separates the two: a name is
// capitalised, "for" in "Thanks for the quick reply." is not. JS regexes have no scoped
// case-insensitive groups like Python's (?i:...), so the closing words are matched
// case-insensitively here and the name is checked case-sensitively by SIGNOFF_TAIL.
const SIGNOFF_HEAD =
  /^\s*(?:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)(?:\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?/i;
// Python's name class [^\W\d_a-zß-öø-ÿ] is "a word char that is not a digit, underscore or
// lowercase letter"; \p{Lu}/\p{Lt}/\p{Lo} is the same intent: a name is capitalised in any
// script (Regards, Łukasz) or written in a script without case (山田).
const SIGNOFF_TAIL = /^[\s,;:!.]*(?:[\p{Lu}\p{Lt}\p{Lo}][\p{L}\p{M}\p{N}_'-]*[\s,.]*){0,3}$/u;
function isEnglishSignoff(line: string): boolean {
  const m = SIGNOFF_HEAD.exec(line);
  return m !== null && SIGNOFF_TAIL.test(line.slice(m[0].length));
}
const SIGNATURE_MARKERS: Array<(line: string) => boolean> = [
  (l) => /^\s*--\s*$/.test(l),
  isEnglishSignoff,
  (l) => /^\s*sent from my (iphone|android|mobile|ipad)/i.test(l),
  (l) =>
    /^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?)[\s,!.]*$/i.test(
      l,
    ),
];
const DEVICE =
  "iphone|ipad|android|ios|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|" +
  "outlook|yahoo|mail|e-?mail|gmail|windows";
const DEVICE_FOOTER = new RegExp(
  "^\\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?)" +
    ` (${DEVICE})( (${DEVICE}|para|for|no|na|\\d+))*|(obter o|get) outlook (para|for) (ios|android))[\\s.!]*$`,
  "i",
);
const DISCLAIMER = new RegExp(
  // English is tied to a disclaimer noun and a disclaimer tail, the way the Portuguese branches
  // below are. The bare word matched any sentence that merely mentioned it, so "Is this
  // confidential?" and "Confidential: I need a refund." were deleted whole and the model was
  // scored on an empty state. `[^.]` rather than `[^.\n]`: a footer wraps, so "are\nconfidential"
  // must still match.
  "(\\b(e-?mail|message|information|communication|transmission|contents?)\\b[^.]{0,60}" +
    "\\bconfidential\\b[^.]{0,60}\\b(intended|solely|addressee|recipient|privileged|" +
    "disclos|unauthori[sz]ed)|" +
    "\\bconfidential\\b[^.]{0,60}\\b(and (may|is) (also )?privileged)|" +
    "if you (have )?received this (e-?mail|message) in error|" +
    "\\b(esta|este) (mensagem|e-?mail|mensaje|correo)\\b[^.]{0,80}(confidencia|sigilos|privilegiad)|" +
    "\\b(uso exclusivo|exclusivamente|únicamente|unicamente)\\b[^.]{0,30}" +
    "(destinatári|destinatari|pessoa|persona|entidade|entidad)|" +
    "\\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\\b[^.]{0,20} por (engano|erro)|" +
    "\\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\\b[^.]{0,20} por error|" +
    "\\bantes de imprimir\\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|" +
    "\\b(meio|medio) ambiente\\b[^.]{0,30}antes de imprimir)",
  "i",
);
const SENTENCE = /(?<=[.!?])\s+/;

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
  for (const line of sentence.split("\n").map((ln) => ln.trim())) {
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
  const parts = paragraph.split(SENTENCE).map((p) => p.trim()).filter(Boolean);
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
  if (text.length > maxChars * 4) text = text.slice(0, maxChars * 4);
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
    if (line.replace(/^\s+/, "").startsWith(">")) continue;
    lines.push(line.replace(/\s+$/, ""));
  }
  let cut = lines.length;
  const start = Math.max(1, Math.min(Math.floor(lines.length * 0.6), lines.length - 8));
  for (let i = start; i < lines.length; i++) {
    const n = lines[i].trim().length;
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
    .split(/\n\s*\n/)
    .map(stripDisclaimer);
  const out = paragraphs
    .map((p) => p.trim())
    .filter(Boolean)
    .join("\n\n")
    .replace(/[ \t]+/g, " ");
  return out.slice(0, maxChars);
}

export function emailState(
  subject: string,
  body: string,
  sender?: string | null,
  clean = true,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  const state: Record<string, unknown> = {
    subject: (subject ?? "").trim(),
    body: clean ? cleanEmailBody(body ?? "") : (body ?? ""),
  };
  if (sender) state["from"] = sender;
  for (const [k, v] of Object.entries(extra ?? {})) {
    if (v !== null && v !== undefined) state[k] = v;
  }
  return state;
}
