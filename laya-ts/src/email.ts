const QUOTE_HEADERS: RegExp[] = [
  /^\s*On .{0,300}wrote:\s*$/i,
  /^\s*Em (?=.*\d).{0,300}escreveu:\s*$/i,
  /^\s*El (?=.*\d).{0,300}escribi[óo]:\s*$/i,
  /^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}/i,
  /^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado))\s*-{2,}/i,
  /^\s*_{8,}\s*$/,
  /^\s*From:\s.+$/i,
  /^\s*De:\s.*[@<]/i,
];
const ATTRIBUTION_TAIL = /^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]):\s*$/i;
const ATTRIBUTION_HEAD = /^\s*(On|Em|El) (?=.*\d)/i;
const HEADER_FROM_NAME = /^\s*De:\s+\S/i;
const HEADER_NEXT = /^\s*(Enviad[oa]( em| el)?:\s|(Data|Fecha):\s.*\d{4})/i;
const SIGNATURE_MARKERS: RegExp[] = [
  /^\s*--\s*$/,
  /^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$/i,
  /^\s*sent from my (iphone|android|mobile|ipad)/i,
  /^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?)[\s,!.]*$/i,
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
  "(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|" +
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
  const text = (body ?? "").replace(/\r\n?/g, "\n").replace(/\\n/g, "\n");
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
      (n <= 40 && SIGNATURE_MARKERS.some((p) => p.test(lines[i]))) ||
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
