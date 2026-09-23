type ScriptRanges = Array<[string, Array<[number, number]>]>;

const SCRIPT_RANGES: ScriptRanges = [
  ["greek", [[0x0370, 0x03ff], [0x1f00, 0x1fff]]],
  ["cyrillic", [[0x0400, 0x052f], [0x2de0, 0x2dff], [0xa640, 0xa69f]]],
  ["armenian", [[0x0530, 0x058f]]],
  ["hebrew", [[0x0590, 0x05ff]]],
  ["arabic", [[0x0600, 0x06ff], [0x0750, 0x077f], [0x08a0, 0x08ff], [0xfb50, 0xfdff], [0xfe70, 0xfeff]]],
  ["devanagari", [[0x0900, 0x097f], [0xa8e0, 0xa8ff]]],
  ["bengali", [[0x0980, 0x09ff]]],
  ["gurmukhi", [[0x0a00, 0x0a7f]]],
  ["gujarati", [[0x0a80, 0x0aff]]],
  ["oriya", [[0x0b00, 0x0b7f]]],
  ["tamil", [[0x0b80, 0x0bff]]],
  ["telugu", [[0x0c00, 0x0c7f]]],
  ["kannada", [[0x0c80, 0x0cff]]],
  ["malayalam", [[0x0d00, 0x0d7f]]],
  ["sinhala", [[0x0d80, 0x0dff]]],
  ["thai", [[0x0e00, 0x0e7f]]],
  ["lao", [[0x0e80, 0x0eff]]],
  ["tibetan", [[0x0f00, 0x0fff]]],
  ["myanmar", [[0x1000, 0x109f]]],
  ["georgian", [[0x10a0, 0x10ff]]],
  ["ethiopic", [[0x1200, 0x137f]]],
  ["khmer", [[0x1780, 0x17ff]]],
  ["hangul", [[0x1100, 0x11ff], [0x3130, 0x318f], [0xac00, 0xd7af]]],
  ["kana", [[0x3040, 0x309f], [0x30a0, 0x30ff], [0x31f0, 0x31ff]]],
  ["han", [[0x3400, 0x4dbf], [0x4e00, 0x9fff], [0xf900, 0xfaff]]],
];

const STOP: Record<string, Set<string>> = {
  en: new Set(["the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
    "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
    "will", "can", "would", "there", "their", "what", "which", "please", "we", "i"]),
  fr: new Set(["le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
    "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
    "et", "du", "au", "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton",
    "ma", "ta", "sa", "mes", "tes", "ses", "ces", "deux", "trois", "très", "bien",
    "tout", "tous", "toute", "fait", "veux", "veut", "peux", "peut", "dois", "doit",
    "merci", "bonjour", "jour", "jours", "mois", "fois", "quand", "comment", "pourquoi",
    "alors", "donc"]),
  de: new Set(["der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
    "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber"]),
  es: new Set(["el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
    "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
    "la", "un", "y", "al", "lo", "le", "les", "su", "mi", "tu", "nos",
    "ni", "dos", "tres", "fue", "fueron", "ser", "tiene", "tienen", "tengo", "puede",
    "pueden", "quiero", "necesito", "hemos", "han", "sobre", "entre", "cuando", "donde",
    "porque", "aunque", "también", "ya", "eso", "esto", "esa", "ese", "nada", "algo",
    "aquí", "hoy", "gracias"]),
  pt: new Set(["os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
    "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
    "o", "e", "na", "nas", "nos", "ao", "aos", "por", "foi", "era", "ser", "sou",
    "tem", "tenho", "pode", "podem", "quero", "preciso", "eu", "meu", "minha", "seu",
    "sua", "isso", "isto", "aqui", "ali", "como", "quando", "onde", "porque", "mais",
    "já", "ainda", "agora", "hoje", "ontem", "dois", "três", "tudo", "nada", "obrigado",
    "olá",
    "você", "vocês", "voce", "voces", "vc", "vcs", "nao", "sao", "ja", "até", "tá", "pra",
    "gostaria", "obrigada", "também", "tambem", "estou", "estamos", "meus", "minhas",
    "nosso", "nossa", "consigo", "cadê", "boa", "tarde", "noite",
    "depois", "antes", "então", "entao", "ninguém", "ninguem", "alguém", "alguem", "nenhum",
    "nenhuma", "estava", "ficou", "fiz", "deu"]),
  it: new Set(["il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
    "questo", "questa", "anche", "come", "più", "sono", "nella", "alla",
    "la", "le", "un", "uno", "una", "e", "ed", "o", "da", "su", "tra", "fra", "mi",
    "ci", "ne", "ho", "hai", "ha", "abbiamo", "avete", "hanno", "era", "stato", "stata",
    "devo", "deve", "devono", "voglio", "vorrei", "mio", "mia", "tuo", "sua", "quando",
    "dove", "perche", "molto", "poco", "sempre", "mai", "già", "ancora", "adesso", "oggi",
    "ieri", "grazie", "ciao", "scusa",
    "nel", "nell", "negli", "sul", "sulla", "sulle", "dal", "dalla", "dallo", "dagli", "dei",
    "delle", "dello", "degli", "agli", "alle", "col"]),
  nl: new Set(["het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
    "door", "maar", "ook", "worden", "deze", "naar", "wordt"]),
  ro: new Set(["și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
    "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
    "acesta", "aceasta", "mi", "ți", "vă", "nu"]),
};

const NON_EN_DIACRITICS = new Set(
  ("àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ" +
    "ăâîșțşţ" +
    "ąćęłńśźż" +
    "čďěňřšťůž" +
    "őű" +
    "ğı" +
    "āēģīķļņūž" +
    "đ").split(""),
);

export const NON_EN_DIACRITIC_RATE = 0.02;

const SHARED_WORDS: Set<string> = (() => {
  const counts = new Map<string, number>();
  for (const words of Object.values(STOP)) {
    for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
  }
  const out = new Set<string>();
  for (const [w, n] of counts) if (n > 1) out.add(w);
  return out;
})();

const WORD_RE = /[^\W\d_]+/gu;
const IDENTIFIER_RE = /[\p{L}\p{N}_-]*(?:[.@][\p{L}\p{N}_-]+)+/gu;
const IS_ALPHA_RE = /\p{L}/u;

function isLatinCp(cp: number): boolean {
  return cp < 0x0250 || (0x1e00 <= cp && cp <= 0x1eff) || (0xff21 <= cp && cp <= 0xff3a) || (0xff41 <= cp && cp <= 0xff5a);
}

export function stateText(state: unknown, maxChars = 4000): string {
  const parts: string[] = [];
  const walk = (v: unknown, d: number): void => {
    if (d > 6 || v == null) return;
    if (typeof v === "string") { parts.push(v); return; }
    if (Array.isArray(v)) { for (const x of v) walk(x, d + 1); return; }
    if (typeof v === "object") { for (const x of Object.values(v as object)) walk(x, d + 1); }
  };
  walk(state, 0);
  return parts.join(" ").slice(0, maxChars);
}

export function detectScript(text: string): string {
  const counts = new Map<string, number>();
  let latin = 0;
  for (const ch of text) {
    if (!IS_ALPHA_RE.test(ch)) continue;
    const cp = ch.codePointAt(0)!;
    if (isLatinCp(cp)) { latin += 1; continue; }
    let found: string | null = null;
    for (const [name, ranges] of SCRIPT_RANGES) {
      if (ranges.some(([lo, hi]) => lo <= cp && cp <= hi)) { found = name; break; }
    }
    counts.set(found ?? "other", (counts.get(found ?? "other") ?? 0) + 1);
  }
  counts.set("latin", latin);
  let total = 0;
  for (const v of counts.values()) total += v;
  if (total === 0) return "unknown";
  let best = "latin";
  let bestN = -1;
  for (const [k, v] of counts) {
    if (v > bestN) { bestN = v; best = k; }
  }
  return best;
}

export function scriptProfile(text: string): Record<string, number> {
  const counts = new Map<string, number>([["latin", 0]]);
  for (const ch of text) {
    if (!IS_ALPHA_RE.test(ch)) continue;
    const cp = ch.codePointAt(0)!;
    if (isLatinCp(cp)) { counts.set("latin", (counts.get("latin") ?? 0) + 1); continue; }
    let found: string | null = null;
    for (const [name, ranges] of SCRIPT_RANGES) {
      if (ranges.some(([lo, hi]) => lo <= cp && cp <= hi)) { found = name; break; }
    }
    const key = found ?? "other";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  let total = 0;
  for (const v of counts.values()) total += v;
  if (!total) return {};
  const out: Record<string, number> = {};
  for (const [k, v] of counts) {
    if (v) out[k] = v / total;
  }
  return out;
}

export interface LatinProfile {
  language: string | null;
  englishHits: number;
  diacriticRate: number;
  looksNonEnglish: boolean;
}

export function latinProfile(text: string): LatinProfile {
  const stripped = text.replace(IDENTIFIER_RE, " ");
  const rawWords = stripped.match(WORD_RE) ?? [];
  const words = rawWords.map((w) => w.toLowerCase());
  const lowered = text.toLowerCase();
  let diac = 0;
  for (const ch of lowered) {
    if (NON_EN_DIACRITICS.has(ch)) diac += 1;
  }
  const diacRate = diac / Math.max(1, lowered.length);
  const nonEnglish = diacRate >= NON_EN_DIACRITIC_RATE;
  if (words.length < 4) {
    return { language: null, englishHits: 0, diacriticRate: diacRate, looksNonEnglish: nonEnglish };
  }
  const scores: Record<string, number> = {};
  for (const [lg, sw] of Object.entries(STOP)) {
    let s = 0;
    for (const w of words) if (sw.has(w)) s += 1;
    scores[lg] = s;
  }
  const en = scores["en"] ?? 0;
  const wordSet = new Set(words);
  let bestLg: string | null = null;
  let best = 0;
  for (const [lg, s] of Object.entries(scores)) {
    if (lg === "en") continue;
    const hasEvidence = [...wordSet].some((w) => STOP[lg].has(w) && !SHARED_WORDS.has(w));
    if (!hasEvidence) continue;
    if (s > best) { best = s; bestLg = lg; }
  }
  let lang: string | null = null;
  if (bestLg && best >= Math.max(2, en + 2)) {
    lang = bestLg;
  } else if (bestLg && nonEnglish && best >= Math.max(2, en)) {
    lang = bestLg;
  } else if (en && !nonEnglish) {
    lang = "en";
  }
  return { language: lang, englishHits: en, diacriticRate: diacRate, looksNonEnglish: nonEnglish };
}

export function guessLatinLanguage(text: string): string | null {
  return latinProfile(text).language;
}

export interface AnalyseResult {
  script: string;
  scriptProfile: Record<string, number>;
  language: string | null;
  isEnglish: boolean;
  languageUndecided: boolean;
  diacriticRate: number;
  nonLatinFraction: number;
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

export function analyse(state: unknown): AnalyseResult {
  const text = stateText(state);
  const prof = scriptProfile(text);
  const script = detectScript(text);
  const nonLatin = prof && Object.keys(prof).length ? round4(1.0 - (prof["latin"] ?? 0.0)) : 0.0;
  if (script === "unknown") {
    return {
      script: "unknown", scriptProfile: prof, language: null,
      isEnglish: true, languageUndecided: true, diacriticRate: 0.0,
      nonLatinFraction: 0.0,
    };
  }
  if (script !== "latin") {
    return {
      script, scriptProfile: prof, language: null,
      isEnglish: false, languageUndecided: true, diacriticRate: 0.0,
      nonLatinFraction: nonLatin,
    };
  }
  const profLat = latinProfile(text);
  const lang = profLat.language;
  const undecided = lang === null;
  const english = lang === "en" || (undecided && !profLat.looksNonEnglish);
  return {
    script: "latin", scriptProfile: prof, language: lang,
    isEnglish: english, languageUndecided: undecided,
    diacriticRate: round4(profLat.diacriticRate),
    nonLatinFraction: nonLatin,
  };
}

export function isEnglish(state: unknown): boolean {
  return analyse(state).isEnglish;
}
