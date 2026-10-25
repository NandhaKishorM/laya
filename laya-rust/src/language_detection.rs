//! Dependency-free language/script detection used to route between Laya checkpoints. Ports
//! `laya/lang.py`.
//!
//! Routing only needs one decision: *is this English Latin text, or is it something the English
//! checkpoint cannot read?* Benchmarks on MASSIVE (14 languages) showed the English checkpoint
//! collapsing to near-random on non-Latin scripts (Hindi 0.100, Korean 0.103, Swahili 0.103, Tamil
//! 0.113 at 20 options, where random is 0.050), while holding up far better on Latin-script
//! languages (French 0.487, Spanish 0.480). So the signal that matters most is *script*, and the
//! secondary signal is whether Latin text is English.
//!
//! Script detection is exact. The Latin-script language guess is a stopword/diacritic heuristic
//! and is explicitly best-effort: pass an explicit model or `lang=` when the language is already
//! known.
//!
//! ## Unicode fidelity
//!
//! Python's `str.isalpha()` tests Unicode general category `L*` (`Lu`, `Ll`, `Lt`, `Lm`, `Lo`)
//! only. Rust's `char::is_alphabetic` also accepts combining marks, which would change
//! [`script_profile`]'s fractions on marked scripts (Hindi matras, for example), so every
//! alphabetic test here goes through [`unicode_general_category::get_general_category`] instead.
//! Python's `[^\W\d_]+` word pattern (`\w` = `isalnum()` + `_`, in Unicode mode) is likewise
//! reproduced with a general-category check rather than a regex crate: `\w` minus digits (`\d` =
//! category `Nd` only) minus `_` leaves letters plus `Nl`/`No` (a numeric-letter or other-number
//! character, but not a plain decimal digit). The other Python patterns this module uses
//! (`_IDENTIFIER`, `_CODE_LINE`, `_JOINED`, `_LETTER_RUN`) and `unicodedata.combining` are
//! hand-written for the same reason, so they agree with Python on every character rather than
//! with the `regex` crate's definition of `\w`.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::char::canonical_combining_class;

/// Namespace for language/script detection. See the module docs for the source this ports.
pub struct LanguageDetection;

/// The full detection result for one piece of state. See [`LanguageDetection::analyse`].
///
/// Marked `#[non_exhaustive]` so a field Python adds later does not break callers again.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LanguageAnalysis {
    /// The dominant script: `"latin"`, `"han"`, `"devanagari"`, ... or `"unknown"` when the text
    /// has no letters at all.
    pub script: String,
    /// Fraction of alphabetic characters belonging to each detected script.
    pub script_profile: HashMap<String, f64>,
    /// Best-effort Latin-script language guess (an ISO-639-1-ish code), or `None` when undecided
    /// or when the script is not Latin.
    pub language: Option<String>,
    /// Whether the English checkpoint can be expected to read this text.
    pub is_english: bool,
    /// Whether [`Self::language`] is `None` because no language could be identified (as opposed
    /// to being `None` because the script itself is non-Latin).
    pub language_undecided: bool,
    /// Fraction of (lowercased) characters that are non-English Latin diacritics.
    pub diacritic_rate: f64,
    /// `1.0 - script_profile["latin"]`, or `0.0` when there is no script profile at all.
    pub non_latin_fraction: f64,
    /// The line or field the segment scan singled out as non-English inside a state that reads
    /// English as a whole, else `None`. A state flipped non-English by the per-value read that
    /// follows the segment scan is reported with this still `None`, exactly as Python does.
    pub mixed_segment: Option<String>,
}

// Unicode blocks that the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
// Order matters: `detect_script`/`script_profile` take the first matching range, and a tie
// between two script counts in `detect_script` is broken by which name was inserted first, which
// is this table's order (for non-Latin scripts) followed by "latin" last.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// A diacritic rate above this is taken as evidence the text is not English, even when no
/// stopword list matches it.
const NON_EN_DIACRITIC_RATE: f64 = 0.02;

/// One accented loanword or proper noun (`café`, `résumé`, `José`) must not alone pull otherwise
/// plain English off the English checkpoint: the rate is measured over every character, so a
/// single `é` in a short sentence clears the floor above. English function words keep their say
/// only through the rescue in [`english_rescued_by_words`]; a rate this high vetoes it.
const ENGLISH_RESCUE_DIACRITIC_RATE: f64 = 0.06;

/// Non-Latin text is not for the English checkpoint even when Latin letters are the plurality: a
/// brand name or order code outvotes the CJK request around it letter for letter. A short
/// message needs a large share to count; a long payload dilutes the share, so there a sentence's
/// worth of letters counts too.
const NON_LATIN_FRACTION: f64 = 0.2;
const NON_LATIN_MIN_FRACTION: f64 = 0.1;
const NON_LATIN_MIN_LETTERS: i64 = 10;

/// An all-caps line must show a matched stopword at least this long before it is believed: the
/// shouted bar of `lang.py` (`_SHOUTED_MIN_STOPWORD`), chosen against three measured corpora.
const SHOUTED_MIN_STOPWORD: usize = 4;
/// ...and a token at least this long (`_SHOUTED_MIN_WORD`): a word rather than an acronym. Also
/// decides whether an all-caps run inside mixed-case text is an acronym worth blanking.
const SHOUTED_MIN_WORD: usize = 5;

/// Letters that ordinary English does not use. This catches a Latin-script language with no
/// stopword list at all (Romanian, Polish, Czech, Turkish, Baltic, ...), which is the difference
/// between routing it to the multilingual checkpoint and silently handing it to the English one.
/// Text is lowercased before matching, so only lowercase forms need listing here.
const NON_EN_DIACRITICS: &str = concat!(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ", // Western European
    "ăâîșțşţ",                         // Romanian
    "ąćęłńśźż",                        // Polish
    "čďěňřšťůž",                       // Czech / Slovak
    "őű",                              // Hungarian
    "ğı",                              // Turkish
    "āēģīķļņūž",                       // Baltic
    "đ",                               // Serbo-Croatian / Vietnamese
    "ə",                               // Azerbaijani
);

// Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
// weighted and a margin is required before calling something non-English. Generated from the
// `_STOP` sets of `laya/lang.py` (sets, so order within a list carries no meaning; the order of
// the lists, in `TABLES`, does).
//
// The Romance lists (fr/es/pt/it) deliberately carry the *unaccented* function words as well as
// the accented ones: a state that lost its accents keeps no diacritic rate for the non-English
// signal to read. `de`/`en` stay out of Spanish and `no` out of Portuguese because they are
// common English tokens. The Swedish list holds words that survive ASCII-normalising ticket
// pipelines, the Bangla list is romanized ("Banglish") and leaves out words that are also
// English, and the Azerbaijani list leaves out `her`, `ne`, `men`, `de` and `ki` (collisions).
const EN_STOP: &[&str] = &[
    "after", "already", "always", "and", "any", "are", "aren", "as", "at", "be", "because", "been",
    "before", "being", "between", "both", "but", "can", "could", "couldn", "didn", "didnt", "does",
    "doesn", "doesnt", "doing", "done", "dont", "down", "during", "each", "ever", "every", "few",
    "for", "from", "has", "hasn", "have", "haven", "havent", "here", "him", "his", "how", "i",
    "if", "in", "into", "is", "isn", "isnt", "it", "ive", "many", "might", "much", "must", "never",
    "nor", "not", "of", "on", "only", "onto", "our", "please", "shall", "she", "should", "some",
    "than", "that", "thats", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "until", "very", "was", "wasn", "wasnt", "we", "were",
    "weren", "what", "whats", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "within", "without", "won", "wont", "would", "wouldn", "yet", "you", "your", "youre",
    "yours",
];

const FR_STOP: &[&str] = &[
    "alors", "au", "aux", "avec", "bien", "bonjour", "ce", "ces", "cette", "comment", "dans",
    "des", "deux", "dois", "doit", "donc", "du", "elle", "elles", "est", "et", "fait", "fois",
    "il", "ils", "je", "jour", "jours", "la", "le", "les", "ma", "mais", "merci", "mes", "mois",
    "mon", "nous", "ont", "ou", "pas", "peut", "peux", "plus", "pour", "pourquoi", "quand", "que",
    "qui", "sa", "ses", "sont", "sur", "ta", "tes", "ton", "tous", "tout", "toute", "trois",
    "très", "tu", "une", "veut", "veux", "vous", "être",
];

const DE_STOP: &[&str] = &[
    "aber", "auch", "auf", "aus", "bei", "bitte", "das", "dem", "den", "der", "dich", "die",
    "diese", "diesen", "dieser", "dieses", "dir", "ein", "eine", "einem", "einen", "einer", "für",
    "gibt", "habe", "haben", "heute", "ich", "im", "in", "ist", "jetzt", "kann", "kannst", "mein",
    "meine", "meinem", "meinen", "meiner", "mich", "mir", "mit", "nach", "nicht", "noch", "oder",
    "sich", "sind", "und", "uns", "von", "wann", "was", "welche", "werden", "wie", "wir", "wird",
    "wo", "wurde", "zu", "zum", "zur",
];

const ES_STOP: &[&str] = &[
    "al", "algo", "aquí", "aunque", "como", "con", "cuando", "del", "donde", "dos", "el", "entre",
    "es", "esa", "ese", "eso", "esta", "este", "esto", "está", "fue", "fueron", "gracias", "han",
    "hay", "hemos", "hoy", "la", "las", "le", "les", "lo", "los", "mi", "muy", "más", "nada",
    "necesito", "ni", "nos", "para", "pero", "por", "porque", "puede", "pueden", "que", "quiero",
    "se", "ser", "sobre", "son", "su", "sus", "también", "tengo", "tiene", "tienen", "todo",
    "tres", "tu", "un", "una", "y", "ya",
];

const PT_STOP: &[&str] = &[
    "agora", "ainda", "alguem", "alguém", "ali", "antes", "ao", "aos", "aqui", "as", "até", "boa",
    "cadê", "com", "como", "consigo", "da", "das", "depois", "deu", "do", "dois", "dos", "e", "em",
    "entao", "então", "era", "esta", "estamos", "estava", "este", "estou", "está", "eu", "ficou",
    "fiz", "foi", "gostaria", "hoje", "isso", "isto", "ja", "já", "mais", "mas", "meu", "meus",
    "minha", "minhas", "muito", "na", "nada", "nao", "nas", "nenhum", "nenhuma", "ninguem",
    "ninguém", "noite", "nos", "nossa", "nosso", "não", "o", "obrigada", "obrigado", "olá", "onde",
    "ontem", "os", "para", "pela", "pelo", "pode", "podem", "por", "porque", "pra", "preciso",
    "quando", "que", "quero", "sao", "se", "ser", "seu", "sou", "sua", "são", "tambem", "também",
    "tarde", "tem", "tenho", "três", "tudo", "tá", "um", "uma", "vc", "vcs", "voce", "voces",
    "você", "vocês", "é",
];

const IT_STOP: &[&str] = &[
    "abbiamo", "adesso", "agli", "alla", "alle", "anche", "ancora", "avete", "che", "ci", "ciao",
    "col", "come", "con", "da", "dagli", "dal", "dalla", "dallo", "degli", "dei", "del", "della",
    "delle", "dello", "deve", "devo", "devono", "di", "dove", "e", "ed", "era", "fra", "già",
    "gli", "grazie", "ha", "hai", "hanno", "ho", "ieri", "il", "la", "le", "lo", "mai", "mi",
    "mia", "mio", "molto", "ne", "negli", "nel", "nell", "nella", "non", "o", "oggi", "per",
    "perche", "più", "poco", "quando", "questa", "questo", "scusa", "sempre", "si", "sono",
    "stata", "stato", "su", "sua", "sul", "sulla", "sulle", "tra", "tuo", "un", "una", "uno",
    "voglio", "vorrei", "è",
];

const NL_STOP: &[&str] = &[
    "aan", "dat", "deze", "door", "een", "het", "is", "maar", "met", "naar", "niet", "ook", "op",
    "te", "van", "voor", "worden", "wordt", "zijn",
];

const SV_STOP: &[&str] = &[
    "appen",
    "aterbetala",
    "aterbetalning",
    "att",
    "behover",
    "behöver",
    "blivit",
    "debiterade",
    "dessa",
    "detta",
    "faktura",
    "fel",
    "från",
    "får",
    "ganger",
    "gång",
    "gånger",
    "gör",
    "göra",
    "hittar",
    "hjalp",
    "hjalpa",
    "hjälp",
    "hjälpa",
    "här",
    "installningarna",
    "installningen",
    "inställningarna",
    "inställningen",
    "inte",
    "jag",
    "kommer",
    "kraschar",
    "kvittot",
    "losenord",
    "lösenord",
    "mig",
    "min",
    "mitt",
    "måste",
    "nar",
    "när",
    "nästa",
    "och",
    "också",
    "om",
    "oppnar",
    "paket",
    "ska",
    "skicka",
    "skulle",
    "sparningen",
    "spårningen",
    "säger",
    "till",
    "tva",
    "två",
    "upp",
    "uppdaterats",
    "utan",
    "vi",
    "vill",
    "är",
    "återbetala",
    "återbetalning",
    "öppnar",
];

const RO_STOP: &[&str] = &[
    "aceasta", "această", "acest", "acesta", "acum", "ale", "care", "dar", "din", "după", "este",
    "foarte", "fost", "fără", "lui", "mi", "nu", "pentru", "până", "sunt", "să", "trebuie",
    "vreau", "vă", "în", "și", "ți",
];

const BN_STOP: &[&str] = &[
    "abar",
    "ajke",
    "akhon",
    "amader",
    "amake",
    "amar",
    "ami",
    "amra",
    "apnake",
    "apnar",
    "apnara",
    "apni",
    "ar",
    "asbe",
    "bhai",
    "bhalo",
    "bolte",
    "bolun",
    "chai",
    "chaina",
    "dhonnobad",
    "dilam",
    "dite",
    "diye",
    "diyechi",
    "dorkar",
    "duibar",
    "ei",
    "eita",
    "ekbar",
    "ekhon",
    "ekhono",
    "ekta",
    "ferot",
    "geche",
    "gese",
    "hobe",
    "hocche",
    "hoise",
    "hoye",
    "hoyeche",
    "hoyni",
    "jabe",
    "jodi",
    "jonno",
    "kalke",
    "keno",
    "keu",
    "kharap",
    "khub",
    "ki",
    "kibhabe",
    "kichu",
    "kintu",
    "kivabe",
    "kobe",
    "kokhon",
    "korbo",
    "korchi",
    "kore",
    "koreche",
    "korechi",
    "koren",
    "korlam",
    "korsi",
    "korte",
    "korun",
    "kothay",
    "koto",
    "lagbe",
    "moddhe",
    "nai",
    "niye",
    "oi",
    "oita",
    "onek",
    "ota",
    "pabo",
    "paini",
    "parben",
    "parbo",
    "parchi",
    "parchina",
    "peyechi",
    "sathe",
    "shathe",
    "shob",
    "shomossa",
    "somossa",
    "tader",
    "tahole",
    "taka",
    "theke",
    "tomake",
    "tomar",
    "tomra",
    "tumi",
    "valo",
];

const AZ_STOP: &[&str] = &[
    "amma", "ancaq", "artiq", "artıq", "bir", "biz", "bu", "cox", "daha", "deyil", "də", "eger",
    "görə", "həm", "hər", "ile", "ilə", "isə", "kimi", "lakin", "mən", "nə", "olan", "olmasa",
    "olub", "onlar", "siz", "sonra", "sən", "ucun", "var", "ve", "və", "yalniz", "yalnız", "yox",
    "yoxdur", "çox", "üçün", "əgər",
];

const SHORT_SWEDISH_WORDS: &[&str] = &[
    "abonnemanget",
    "aterbetalning",
    "atkomst",
    "behover",
    "behöver",
    "betalningen",
    "fakturan",
    "felmeddelande",
    "glomt",
    "glömt",
    "inloggningen",
    "installningen",
    "inställningen",
    "kvitto",
    "kvittot",
    "losenord",
    "lösenord",
    "sparningen",
    "spårningen",
    "återbetalning",
    "åtkomst",
];

const NORDIC_OVERLAP_WORDS: &[&str] = &[
    "får", "hej", "ja", "jo", "kommer", "mig", "min", "nej", "om", "skulle", "tack", "vi",
];

/// The stop lists in `_STOP` iteration order (`en` first): the order is load-bearing for
/// `latin_profile`'s tie-break, which keeps the earlier language.
struct StopTables {
    lists: Vec<(&'static str, HashSet<&'static str>)>,
    /// Words that more than one list claims (plus the Swedish-Danish overlap words): a shared
    /// word may not name a winner by itself, though it still counts toward the total of a
    /// language that also matched a word of its own.
    shared: HashSet<&'static str>,
    /// English function words no other list holds. They alone carry the English rescue.
    en_only: HashSet<&'static str>,
    /// Function words of another language that are also ordinary English words: a repeat is
    /// evidence of the foreign language exactly once.
    collision: HashSet<&'static str>,
    nordic_overlap: HashSet<&'static str>,
    short_swedish: HashSet<&'static str>,
}

static TABLES: LazyLock<StopTables> = LazyLock::new(|| {
    let raw: [(&'static str, &'static [&'static str]); 11] = [
        ("en", EN_STOP),
        ("fr", FR_STOP),
        ("de", DE_STOP),
        ("es", ES_STOP),
        ("pt", PT_STOP),
        ("it", IT_STOP),
        ("nl", NL_STOP),
        ("sv", SV_STOP),
        ("ro", RO_STOP),
        ("bn", BN_STOP),
        ("az", AZ_STOP),
    ];
    let lists: Vec<(&'static str, HashSet<&'static str>)> = raw
        .iter()
        .map(|(code, words)| (*code, words.iter().copied().collect()))
        .collect();
    let mut membership: HashMap<&'static str, usize> = HashMap::new();
    for (_, set) in &lists {
        for word in set {
            *membership.entry(word).or_insert(0) += 1;
        }
    }
    let nordic_overlap: HashSet<&'static str> = NORDIC_OVERLAP_WORDS.iter().copied().collect();
    let mut shared: HashSet<&'static str> = membership
        .into_iter()
        .filter(|&(_, n)| n > 1)
        .map(|(w, _)| w)
        .collect();
    shared.extend(nordic_overlap.iter().copied());
    let en_only = lists[0].1.difference(&shared).copied().collect();
    let foreign: HashSet<&'static str> = lists[1..]
        .iter()
        .flat_map(|(_, set)| set.iter().copied())
        .collect();
    let collision = ["come", "son", "do", "care", "todo", "im", "per", "plus"]
        .into_iter()
        .filter(|w| foreign.contains(w))
        .collect();
    StopTables {
        lists,
        shared,
        en_only,
        collision,
        nordic_overlap,
        short_swedish: SHORT_SWEDISH_WORDS.iter().copied().collect(),
    }
});

fn stop_set(code: &str) -> Option<&'static HashSet<&'static str>> {
    TABLES
        .lists
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, set)| set)
}

// ---------------------------------------------------------------------------------------------
// Python-compatible character classes. `regex`'s `\w`, `\s` and `\b` differ from `re`'s on marks,
// joiners and a few control characters, so the patterns `lang.py` runs are hand-written over
// these predicates.
// ---------------------------------------------------------------------------------------------

/// `str.isspace()`: Unicode White_Space plus U+001C..U+001F (bidi class B/S), which `re`'s `\s`
/// accepts and the `regex` crate's does not.
pub(crate) fn py_is_space(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

/// `str.strip()`.
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(py_is_space)
}

/// `\d`: decimal digits (`Nd`) only.
fn is_decimal(ch: char) -> bool {
    get_general_category(ch) == GeneralCategory::DecimalNumber
}

/// `[^\W_]`: `isalnum()` -- letters and every kind of number, but not the underscore.
fn is_alnum(ch: char) -> bool {
    LanguageDetection::is_alpha(ch)
        || matches!(
            get_general_category(ch),
            GeneralCategory::DecimalNumber
                | GeneralCategory::LetterNumber
                | GeneralCategory::OtherNumber
        )
}

/// `\w`: [`is_alnum`] plus the underscore.
pub(crate) fn is_word_char(ch: char) -> bool {
    ch == '_' || is_alnum(ch)
}

/// `[^\W\d_]`: a letter, or a numeral that is not a plain decimal digit.
pub(crate) fn is_letter_word(ch: char) -> bool {
    is_alnum(ch) && !is_decimal(ch)
}

fn is_titlecase(ch: char) -> bool {
    get_general_category(ch) == GeneralCategory::TitlecaseLetter
}

/// `ch.isupper()` for a one-character string.
pub(crate) fn char_is_upper(ch: char) -> bool {
    ch.is_uppercase() && !ch.is_lowercase() && !is_titlecase(ch)
}

/// `ch.islower()` for a one-character string.
fn char_is_lower(ch: char) -> bool {
    ch.is_lowercase() && !ch.is_uppercase() && !is_titlecase(ch)
}

/// `s.isupper()`: at least one uppercase letter and no lowercase or titlecase one.
fn str_is_upper(s: &str) -> bool {
    let mut cased = false;
    for ch in s.chars() {
        if ch.is_lowercase() || is_titlecase(ch) {
            return false;
        }
        if ch.is_uppercase() {
            cased = true;
        }
    }
    cased
}

/// The first `n` characters of `s` (Python's `s[:n]`).
pub(crate) fn prefix_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Maximal runs of `[^\W\d_]` in `text` (`_WORD.findall`).
fn word_runs(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in text.char_indices() {
        if is_letter_word(ch) {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            words.push(&text[s..idx]);
        }
    }
    if let Some(s) = start {
        words.push(&text[s..]);
    }
    words
}

/// `_IDENTIFIER.sub(" ", text)`: `(?<![\w-])[\w-]*(?:[.@][\w-]+)+`. A token whose dot or `@` joins
/// word characters is an identifier, not prose (`github.com`, `user@acme.com`, `v1.2.3`). A
/// sentence-final period keeps its word: the pattern needs word characters on both sides.
fn replace_identifiers(text: &str) -> String {
    let is_ident = |c: char| is_word_char(c) || c == '-';
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < n {
        // The lookbehind only lets a match start at the head of a run, which is also what keeps
        // this linear: a shorter `[\w-]*` could never be followed by `[.@]`.
        if i == 0 || !is_ident(chars[i - 1]) {
            let mut j = i;
            while j < n && is_ident(chars[j]) {
                j += 1;
            }
            let mut groups = 0;
            while j < n && (chars[j] == '.' || chars[j] == '@') {
                let mut k = j + 1;
                while k < n && is_ident(chars[k]) {
                    k += 1;
                }
                if k == j + 1 {
                    break;
                }
                j = k;
                groups += 1;
            }
            if groups > 0 {
                out.push(' ');
                i = j;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `_CODE_LINE.search`: `[=;{}\[\]]|\w\(` -- a line carrying code syntax is not prose.
fn has_code_syntax(line: &str) -> bool {
    let mut prev_word = false;
    for ch in line.chars() {
        if matches!(ch, '=' | ';' | '{' | '}' | '[' | ']') || (ch == '(' && prev_word) {
            return true;
        }
        prev_word = is_word_char(ch);
    }
    false
}

/// `_JOINED.search(token)`: `[^\W_][._/\\][^\W_]` -- a name, path or compound (`Nav/Com`, `OS/2`,
/// `C:\DOS\mode`), not a sentence.
fn is_joined_token(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    chars
        .windows(3)
        .any(|w| is_alnum(w[0]) && matches!(w[1], '.' | '_' | '/' | '\\') && is_alnum(w[2]))
}

/// `_LETTER_RUN.finditer`: maximal runs of at least two `[^\W\d_]` characters, as byte ranges.
fn letter_runs(text: &str) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start: Option<(usize, usize)> = None; // (byte offset, char count so far)
    for (idx, ch) in text.char_indices() {
        if is_letter_word(ch) {
            start = Some(match start {
                Some((s, n)) => (s, n + 1),
                None => (idx, 1),
            });
        } else if let Some((s, n)) = start.take()
            && n >= 2
        {
            runs.push((s, idx));
        }
    }
    if let Some((s, n)) = start
        && n >= 2
    {
        runs.push((s, text.len()));
    }
    runs
}

/// `_LETTER_RUN.sub(lambda m: " " if m.group().isupper() else m.group(), text)`.
fn blank_upper_runs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for (s, e) in letter_runs(text) {
        if str_is_upper(&text[s..e]) {
            out.push_str(&text[last..s]);
            out.push(' ');
            last = e;
        }
    }
    out.push_str(&text[last..]);
    out
}

impl LanguageDetection {
    /// Whether `ch` is alphabetic under Python's `str.isalpha()` rule: Unicode general category
    /// `Lu`, `Ll`, `Lt`, `Lm` or `Lo`. Unlike `char::is_alphabetic`, this excludes combining
    /// marks.
    pub(crate) fn is_alpha(ch: char) -> bool {
        matches!(
            get_general_category(ch),
            GeneralCategory::UppercaseLetter
                | GeneralCategory::LowercaseLetter
                | GeneralCategory::TitlecaseLetter
                | GeneralCategory::ModifierLetter
                | GeneralCategory::OtherLetter
        )
    }

    /// First script whose range table contains `cp`, if any.
    fn script_for_codepoint(cp: u32) -> Option<&'static str> {
        for (name, ranges) in SCRIPT_RANGES {
            if ranges.iter().any(|&(lo, hi)| cp >= lo && cp <= hi) {
                return Some(name);
            }
        }
        None
    }

    /// Whether `cp` counts as Latin: Basic Latin through the IPA Extensions (below U+02B0),
    /// Latin Extended Additional, and the fullwidth Latin letters.
    fn is_latin_codepoint(cp: u32) -> bool {
        cp < 0x02B0
            || (0x1E00..=0x1EFF).contains(&cp)
            || (0xFF21..=0xFF3A).contains(&cp)
            || (0xFF41..=0xFF5A).contains(&cp)
    }

    /// Collect the string leaves of a state (str / object / array), so detection sees real
    /// content. Depths beyond 6 are dropped, and keys are ignored (they are usually English).
    fn iter_text<'a>(state: &'a serde_json::Value, depth: u32, out: &mut Vec<&'a str>) {
        if depth > 6 {
            return;
        }
        match state {
            serde_json::Value::String(s) => out.push(s),
            serde_json::Value::Object(map) => {
                for v in map.values() {
                    Self::iter_text(v, depth + 1, out);
                }
            }
            serde_json::Value::Array(items) => {
                for v in items {
                    Self::iter_text(v, depth + 1, out);
                }
            }
            _ => {}
        }
    }

    fn leaves(state: &serde_json::Value) -> Vec<&str> {
        let mut out = Vec::new();
        Self::iter_text(state, 0, &mut out);
        out
    }

    /// Flatten a state into the text used for detection (keys are ignored: they are usually
    /// English). `max_chars` truncates by Unicode scalar value, matching Python's `[:max_chars]`
    /// string slice.
    pub fn state_text(state: &serde_json::Value, max_chars: usize) -> String {
        // lang.py:state_text stops adding leaves once the budget (which counts each joining
        // space) is spent, so a leaf that ends exactly at the budget gets no trailing space.
        let mut parts: Vec<&str> = Vec::new();
        let mut budget = max_chars as i64;
        for leaf in Self::leaves(state) {
            if budget <= 0 {
                break;
            }
            let len = leaf.chars().count() as i64;
            if len > budget {
                parts.push(prefix_chars(leaf, budget as usize));
                break;
            }
            parts.push(leaf);
            budget -= len + 1;
        }
        prefix_chars(&parts.join(" "), max_chars).to_string()
    }

    /// [`Self::state_text`] with Python's default `max_chars=4000`.
    pub fn state_text_default(state: &serde_json::Value) -> String {
        Self::state_text(state, 4000)
    }

    /// The state Python's `_iter_text` sees for a `bytes`/`bytearray` state: the bytes decoded as
    /// strict UTF-8, or no text at all when they are not valid UTF-8 (`lang.py:253-255`).
    fn bytes_state(bytes: &[u8]) -> serde_json::Value {
        serde_json::Value::String(std::str::from_utf8(bytes).unwrap_or("").to_string())
    }

    /// [`Self::state_text`] for a `bytes` state, as Python accepts one: strict UTF-8, and invalid
    /// UTF-8 reads as no text (`""`).
    pub fn state_text_bytes(state: &[u8], max_chars: usize) -> String {
        Self::state_text(&Self::bytes_state(state), max_chars)
    }

    /// [`Self::analyse`] for a `bytes` state, as Python accepts one: strict UTF-8, and invalid
    /// UTF-8 reads as no text.
    pub fn analyse_bytes(state: &[u8]) -> LanguageAnalysis {
        Self::analyse(&Self::bytes_state(state))
    }

    /// [`Self::is_english`] for a `bytes` state.
    pub fn is_english_bytes(state: &[u8]) -> bool {
        Self::analyse_bytes(state).is_english
    }

    /// Dominant script of `text`: `"latin"`, `"han"`, `"devanagari"`, ... or `"unknown"` if there
    /// are no letters.
    ///
    /// Ties are broken by first-inserted order: the non-Latin scripts in the order they first
    /// appear in `text`, then `"latin"` last, so Latin never wins a tie against a non-Latin
    /// script present in the same text.
    pub fn detect_script(text: &str) -> String {
        ScriptCounts::of(text).dominant().to_string()
    }

    /// Fraction of alphabetic characters belonging to each detected script.
    pub fn script_profile(text: &str) -> HashMap<String, f64> {
        ScriptCounts::of(text).profile()
    }

    /// Evidence behind the Latin-script language guess: `language` (may be `None` when
    /// undecided), `english_hits`, `diacritic_rate` and `looks_non_english`. [`Self::analyse`]
    /// needs the evidence and not just the verdict, because "undecided" and "English" are
    /// different answers and only one of them is safe to send to the English checkpoint.
    ///
    /// A non-English language is only named when it matched at least one word that no other list
    /// claims: shared function words alone (`la`, `e`, `o`) identify no particular language.
    /// `looks_non_english` can be true for non-English diacritics or an overlapping
    /// Swedish-Danish marker even when no language is named.
    pub fn latin_profile(text: &str) -> LatinProfile {
        let p = latin_profile_inner(text);
        LatinProfile {
            language: p.language.map(str::to_string),
            english_hits: p.english_hits,
            diacritic_rate: p.diacritic_rate,
            looks_non_english: p.looks_non_english,
        }
    }

    /// Best-effort language code for Latin-script text, or `None` when undecided.
    ///
    /// Scores function-word hits per language and requires the winner to beat English by a
    /// margin and to have matched at least one word of its own, so ordinary English is never
    /// misrouted and a word of several languages at once names none of them. Short inputs
    /// usually return `None` on purpose.
    pub fn guess_latin_language(text: &str) -> Option<String> {
        Self::latin_profile(text).language
    }

    /// Full detection result for a state. See [`LanguageAnalysis`].
    ///
    /// String values are what get read. A state that reads English as a whole is then checked
    /// line by line and field by field, because the English part is usually the longer one and
    /// outvotes a short message in another language (a Portuguese ticket with an English stack
    /// trace); one non-English value is enough to move the state to the multilingual checkpoint.
    pub fn analyse(state: &serde_json::Value) -> LanguageAnalysis {
        let leaves = Self::leaves(state);
        let mut result = analyse_text(&Self::state_text_default(state));
        // a single line has no other part to be outvoted by, and was just read whole
        if result.script == "latin"
            && result.is_english
            && (leaves.len() > 1 || leaves.iter().any(|leaf| leaf.contains('\n')))
            && let Some((lang, mixed)) = non_english_segment(&leaves, 4000)
        {
            result.language = Some(lang.to_string());
            result.is_english = false;
            result.language_undecided = false;
            result.mixed_segment = Some(mixed);
        }
        // A plain string was just read whole. A structured state can still hide a message past
        // the segment cap, or in a script `latin_profile` does not name.
        if matches!(
            state,
            serde_json::Value::String(_) | serde_json::Value::Null
        ) || !result.is_english
        {
            return result;
        }
        let mut best_n: i64 = -1;
        let mut best: Option<LanguageAnalysis> = None;
        for leaf in &leaves {
            let Some(det) = leaf_non_english(leaf) else {
                continue;
            };
            let n_alpha = prefix_chars(leaf, 4000)
                .chars()
                .filter(|&c| Self::is_alpha(c))
                .count() as i64;
            if n_alpha > best_n {
                best_n = n_alpha;
                best = Some(det);
            }
        }
        if let Some(best) = best {
            result.language = best.language;
            result.is_english = false;
            result.language_undecided = best.language_undecided;
        }
        result
    }

    /// True when the English checkpoint can be expected to read this state.
    pub fn is_english(state: &serde_json::Value) -> bool {
        Self::analyse(state).is_english
    }
}

/// The evidence [`LanguageDetection::latin_profile`] gathers for a Latin-script language guess.
#[derive(Debug, Clone, PartialEq)]
pub struct LatinProfile {
    /// The guessed language code, or `None` when undecided.
    pub language: Option<String>,
    /// How many English stopword hits the text had.
    pub english_hits: i64,
    /// Fraction of (lowercased) characters that are non-English Latin diacritics.
    pub diacritic_rate: f64,
    /// Whether the text looks non-English on its own evidence: a diacritic rate at or above the
    /// threshold, or a Swedish-Danish overlap marker with no language named.
    pub looks_non_english: bool,
}

struct LatinProfileInner {
    language: Option<&'static str>,
    english_hits: i64,
    diacritic_rate: f64,
    looks_non_english: bool,
}

/// Per-script letter counts of one text, in `lang.py`'s `_script_counts` insertion order.
struct ScriptCounts {
    /// Named scripts (and `"other"`, for letters no range claims) in first-seen order.
    named: Vec<(&'static str, i64)>,
    latin: i64,
}

impl ScriptCounts {
    fn of(text: &str) -> Self {
        let mut named: Vec<(&'static str, i64)> = Vec::new();
        let mut latin = 0;
        for ch in text.chars() {
            if !LanguageDetection::is_alpha(ch) {
                continue;
            }
            let cp = ch as u32;
            if LanguageDetection::is_latin_codepoint(cp) {
                latin += 1; // Latin, IPA Extensions, Extended Additional, fullwidth
                continue;
            }
            // A letter no range claims is counted under "other": text in an unlisted script must
            // stay visible and non-Latin, never be reported as "unknown" and handed to English.
            let name = LanguageDetection::script_for_codepoint(cp).unwrap_or("other");
            match named.iter_mut().find(|(n, _)| *n == name) {
                Some(entry) => entry.1 += 1,
                None => named.push((name, 1)),
            }
        }
        Self { named, latin }
    }

    fn total(&self) -> i64 {
        self.latin + self.named.iter().map(|(_, c)| c).sum::<i64>()
    }

    /// `_script_from_counts`: the first script with the highest count; Latin is last, so a named
    /// script wins a tie against it.
    fn dominant(&self) -> &'static str {
        if self.total() == 0 {
            return "unknown";
        }
        let mut best: Option<(&'static str, i64)> = None;
        for &(name, count) in self
            .named
            .iter()
            .chain(std::iter::once(&("latin", self.latin)))
        {
            if best.is_none_or(|(_, b)| count > b) {
                best = Some((name, count));
            }
        }
        best.map_or("unknown", |(name, _)| name)
    }

    fn profile(&self) -> HashMap<String, f64> {
        let total = self.total();
        if total == 0 {
            return HashMap::new();
        }
        let mut out = HashMap::new();
        if self.latin != 0 {
            out.insert("latin".to_string(), self.latin as f64 / total as f64);
        }
        for &(name, count) in &self.named {
            if count != 0 {
                out.insert(name.to_string(), count as f64 / total as f64);
            }
        }
        out
    }

    /// `max(s for s in prof if s != "latin", key=prof.get)`: the first non-Latin script with the
    /// highest count.
    fn dominant_non_latin(&self) -> Option<&'static str> {
        let mut best: Option<(&'static str, i64)> = None;
        for &(name, count) in &self.named {
            if best.is_none_or(|(_, b)| count > b) {
                best = Some((name, count));
            }
        }
        best.map(|(name, _)| name)
    }
}

fn round4(x: f64) -> f64 {
    crate::calibration::Calibration::round4(x)
}

/// Whether the non-Latin letters of a Latin-plurality text are enough to carry the state.
fn non_latin_carries(non_latin: f64, n_non_latin: i64) -> bool {
    non_latin >= NON_LATIN_FRACTION
        || (non_latin >= NON_LATIN_MIN_FRACTION && n_non_latin >= NON_LATIN_MIN_LETTERS)
}

/// `_script_of`: the named non-Latin script of one character, or `None` for Latin and for
/// characters no range claims. (Not restricted to letters: script punctuation joins its run.)
fn script_of(ch: char) -> Option<&'static str> {
    let cp = ch as u32;
    if cp < 0x0250
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
    {
        return None;
    }
    LanguageDetection::script_for_codepoint(cp)
}

/// `_non_latin_words`: non-Latin runs that read as words rather than as annotation inside
/// English prose. A symbol (`Set α to 0.05`, one letter), a proper name (`Дмитрий Петрович`,
/// capitalised) and a pronunciation (`[vlɐˈdʲimʲɪr]`, which no range claims) are excluded. A
/// character with a canonical combining class belongs to the letter before it and never splits a
/// word.
fn non_latin_words(text: &str) -> Vec<String> {
    let mut runs: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut script: Option<&'static str> = None;
    for ch in text.chars() {
        if canonical_combining_class(ch) != 0 {
            continue;
        }
        let s = script_of(ch);
        if s.is_some() && s == script {
            cur.push(ch);
            continue;
        }
        if !cur.is_empty() {
            runs.push(std::mem::take(&mut cur));
        }
        if s.is_some() {
            cur.push(ch);
        }
        script = s;
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs.into_iter()
        .filter(|w| w.chars().count() >= 2 && !w.chars().next().is_some_and(char_is_upper))
        .collect()
}

/// `_english_rescued_by_words`: whether plain-English function words outvote a marginal
/// diacritic rate. English still wins when it shows at least two distinct function words no
/// other list holds and at most one word carrying a non-English letter. A rate well above
/// [`ENGLISH_RESCUE_DIACRITIC_RATE`] vetoes regardless.
fn english_rescued_by_words(word_set: &HashSet<&str>, diac_rate: f64) -> bool {
    if diac_rate >= ENGLISH_RESCUE_DIACRITIC_RATE {
        return false;
    }
    if word_set.intersection(&TABLES.en_only).count() < 2 {
        return false;
    }
    word_set
        .iter()
        .filter(|w| w.chars().any(|c| NON_EN_DIACRITICS.contains(c)))
        .count()
        <= 1
}

fn latin_profile_inner(text: &str) -> LatinProfileInner {
    // 'İ'.lower() is 'i' + a combining dot, which matches no word list.
    let cleaned = replace_identifiers(text).replace('İ', "i").to_lowercase();
    let words = word_runs(&cleaned);
    let lowered = text.to_lowercase();
    let diac = lowered
        .chars()
        .filter(|c| NON_EN_DIACRITICS.contains(*c))
        .count();
    let diac_rate = diac as f64 / (lowered.chars().count().max(1) as f64);
    let non_english = diac_rate >= NON_EN_DIACRITIC_RATE;
    let word_set: HashSet<&str> = words.iter().copied().collect();
    let tables = &*TABLES;
    let nordic_overlap =
        !word_set.is_disjoint(&tables.nordic_overlap) && word_set.is_disjoint(&tables.en_only);
    if words.len() > 1 && words.len() < 4 && !word_set.is_disjoint(&tables.short_swedish) {
        return LatinProfileInner {
            language: Some("sv"),
            english_hits: 0,
            diacritic_rate: diac_rate,
            looks_non_english: non_english,
        };
    }
    if words.len() < 4 {
        return LatinProfileInner {
            language: None,
            english_hits: 0,
            diacritic_rate: diac_rate,
            looks_non_english: non_english || nordic_overlap,
        };
    }

    // A collision word counts once however often it repeats; every other word counts its hits.
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for w in &words {
        *counts.entry(w).or_insert(0) += 1;
    }
    let score_of = |set: &HashSet<&'static str>| -> i64 {
        counts
            .iter()
            .filter(|(w, _)| set.contains(*w))
            .map(|(w, n)| if tables.collision.contains(w) { 1 } else { *n })
            .sum()
    };
    let en = score_of(&tables.lists[0].1);

    // Only a language that matched at least one word no other list claims may be named. Without
    // that the top score can be pure overlap -- `la` and `e` in Romanian text made Italian the
    // winner -- which is a guess dressed as a detection. Such a language is dropped from the
    // running rather than merely losing the tie. Ties keep the earlier language (`max()` returns
    // the first maximum in `_STOP` order).
    let mut best_lg: Option<&'static str> = None;
    let mut best: i64 = 0;
    for (code, set) in tables.lists.iter().skip(1) {
        if !word_set
            .iter()
            .any(|w| set.contains(w) && !tables.shared.contains(w))
        {
            continue;
        }
        let score = score_of(set);
        if best_lg.is_none() || score > best {
            best_lg = Some(code);
            best = score;
        }
    }

    let mut language: Option<&'static str> = None;
    if best_lg.is_some() && best >= 2i64.max(en + 2) {
        // a non-English language needs a clear margin over English function words
        language = best_lg;
    } else if best_lg == Some("sv")
        && word_set.contains("inte")
        && word_set.contains("kan")
        && matches!(words[0], "kan" | "jag" | "vi")
        && en <= 1
    {
        // Short login requests such as "kan inte logga in" carry a distinctive Swedish phrase
        // but also one English-shaped token (`in`). `kan` alone is not Swedish: it is common in
        // Danish and Norwegian too.
        language = best_lg;
    } else if best_lg.is_some() && non_english && best >= 2i64.max(en) {
        // Needs two hits here too: one shared function word ("para" in Turkish text) named
        // Spanish on the strength of the diacritics alone, which is a guess dressed as a
        // detection.
        language = best_lg;
    } else if en > 0 && (!non_english || english_rescued_by_words(&word_set, diac_rate)) {
        language = Some("en");
    }
    LatinProfileInner {
        language,
        english_hits: en,
        diacritic_rate: diac_rate,
        looks_non_english: non_english || (language.is_none() && nordic_overlap),
    }
}

/// `_shouted`: true for a *cased* segment written entirely in capitals. The `isupper` half is
/// not redundant: a caseless script (Devanagari, Bengali, CJK) has no lowercase either, and the
/// bar belongs to cased text.
fn shouted(text: &str) -> bool {
    text.chars().any(char_is_upper) && !text.chars().any(char_is_lower)
}

/// `_shouted_evidence`: whether an all-caps line's evidence for `lang` is more than
/// acronym-shaped tokens.
fn shouted_evidence(tokens: &[&str], lang: &str, diacritic_rate: f64) -> bool {
    if !tokens.iter().any(|t| t.chars().count() >= SHOUTED_MIN_WORD) {
        return false;
    }
    if diacritic_rate >= NON_EN_DIACRITIC_RATE {
        return true;
    }
    let Some(stop) = stop_set(lang) else {
        return false;
    };
    tokens
        .iter()
        .map(|t| t.to_lowercase())
        .any(|w| stop.contains(w.as_str()) && w.chars().count() >= SHOUTED_MIN_STOPWORD)
}

/// `_named_prose_language`: the language code for one non-code line, or `None` when it does not
/// name a foreign language. Four words, a language `latin_profile` will name, and two
/// *different* words of that language; acronyms and slash compounds are not words, and a segment
/// in all capitals must also clear [`shouted_evidence`].
fn named_prose_language(segment: &str) -> Option<&'static str> {
    if py_strip(segment).is_empty() || has_code_syntax(segment) {
        return None;
    }
    let mut prose = segment
        .split(py_is_space)
        .filter(|tok| !tok.is_empty() && !is_joined_token(tok))
        .collect::<Vec<_>>()
        .join(" ");
    let is_shouted = shouted(&prose);
    if !is_shouted {
        prose = blank_upper_runs(&prose);
    }
    let tokens = word_runs(&prose);
    if tokens.len() < 4 {
        return None;
    }
    let prof = latin_profile_inner(&prose);
    let lang = prof.language?;
    if lang == "en" {
        return None;
    }
    if is_shouted && !shouted_evidence(&tokens, lang, prof.diacritic_rate) {
        return None;
    }
    let stop = stop_set(lang)?;
    let distinct: HashSet<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
    if distinct
        .iter()
        .filter(|w| stop.contains(w.as_str()))
        .count()
        < 2
    {
        return None;
    }
    Some(lang)
}

/// `_non_english_segment`: the first line or field that, read on its own, is named a non-English
/// language -- `(language, segment)`. Reads at most `max_chars` characters in all.
fn non_english_segment(leaves: &[&str], max_chars: usize) -> Option<(&'static str, String)> {
    let mut seen = 0usize;
    for leaf in leaves {
        for seg in leaf.split('\n') {
            if seen >= max_chars {
                return None;
            }
            let seg = prefix_chars(seg, max_chars - seen);
            seen += seg.chars().count();
            if let Some(lang) = named_prose_language(seg) {
                return Some((lang, py_strip(seg).to_string()));
            }
        }
    }
    None
}

/// `_analyse_text`: the detection result for one already-flattened string. Does not look for a
/// foreign line inside mostly-English text; [`LanguageDetection::analyse`] does that.
fn analyse_text(text: &str) -> LanguageAnalysis {
    let counts = ScriptCounts::of(text);
    let total = counts.total();
    let prof = counts.profile();
    let mut script = counts.dominant();
    let non_latin = if total > 0 {
        round4(1.0 - counts.latin as f64 / total as f64)
    } else {
        0.0
    };
    let n_non_latin = (non_latin * total as f64).round_ties_even() as i64;
    if script == "latin"
        && !non_latin_words(text).is_empty()
        && non_latin_carries(non_latin, n_non_latin)
        && let Some(promoted) = counts.dominant_non_latin()
    {
        script = promoted;
    }
    if script == "unknown" {
        return LanguageAnalysis {
            script: "unknown".to_string(),
            script_profile: prof,
            language: None,
            is_english: true,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: 0.0,
            mixed_segment: None,
        };
    }
    if script != "latin" {
        return LanguageAnalysis {
            script: script.to_string(),
            script_profile: prof,
            language: None,
            is_english: false,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: non_latin,
            mixed_segment: None,
        };
    }
    let prof_lat = latin_profile_inner(text);
    let mut lang = prof_lat.language;
    // The acronym bar belongs here too, not only in the segment scan: a state that is nothing
    // but an acronym line (`MON DES EST LA`) must not be named foreign outright. Vetoing leaves
    // the text undecided, so `looks_non_english` still decides it. Neither bar applies when the
    // state is carried by a caseless script: they take evidence away from the Latin letters, and
    // `looks_non_english` reads Latin diacritics only.
    if let Some(named) = lang
        && named != "en"
        && !non_latin_carries(non_latin, n_non_latin)
    {
        if shouted(text) {
            if !shouted_evidence(&word_runs(text), named, prof_lat.diacritic_rate) {
                lang = None;
            }
        } else {
            // Mixed-case text: re-take the verdict without the all-caps runs, but only while
            // every one of them is short enough to be an acronym -- a run of
            // `SHOUTED_MIN_WORD` letters is a word, and a word is not blanked out of its own
            // sentence. `looks_non_english` and `diacritic_rate` stay measured on the original
            // text.
            let caps: Vec<&str> = letter_runs(text)
                .into_iter()
                .map(|(s, e)| &text[s..e])
                .filter(|run| str_is_upper(run))
                .collect();
            if !caps.is_empty() && !caps.iter().any(|c| c.chars().count() >= SHOUTED_MIN_WORD) {
                let blanked = blank_upper_runs(text);
                if blanked != text {
                    lang = latin_profile_inner(&blanked).language;
                }
            }
        }
    }
    // Undecided is not English. Treating it as English sent every Latin-script language with no
    // stopword list here to the checkpoint that cannot read it, silently. When nothing
    // identifies the language, non-English letters or a shared Swedish-Danish marker can still
    // prefer the multilingual checkpoint; text with neither signal goes to the English one.
    let undecided = lang.is_none();
    let english = lang == Some("en") || (undecided && !prof_lat.looks_non_english);
    LanguageAnalysis {
        script: "latin".to_string(),
        script_profile: prof,
        language: lang.map(str::to_string),
        is_english: english,
        language_undecided: undecided,
        diacritic_rate: round4(prof_lat.diacritic_rate),
        non_latin_fraction: non_latin,
        mixed_segment: None,
    }
}

/// `_leaf_non_english`: a string value that is itself not safe for the English checkpoint, else
/// `None`. A one-word name and a capitalised non-Latin name stay out, as do code lines,
/// acronyms and slash compounds. Each line is capped at 4000 characters; a long earlier field
/// does not consume the budget of the next one.
fn leaf_non_english(leaf: &str) -> Option<LanguageAnalysis> {
    let mut best_n: i64 = -1;
    let mut best: Option<LanguageAnalysis> = None;
    for line in leaf.split('\n') {
        // A line this short cannot be selected: each branch below needs four words (three
        // separators, seven characters) or `NON_LATIN_MIN_LETTERS` letters.
        if line.chars().count() < 7 {
            continue;
        }
        let sample = prefix_chars(line, 4000);
        if py_strip(sample).is_empty() || has_code_syntax(sample) {
            continue;
        }
        let det = analyse_text(sample);
        if det.is_english {
            continue;
        }
        // A named language still has to survive `named_prose_language`, but a veto is not a
        // verdict of English: a line carrying non-English letters that no stopword list can
        // name falls through to the diacritic branch.
        let named = det.language.as_deref().is_some_and(|l| l != "en")
            && named_prose_language(sample).is_some();
        let n_alpha = sample
            .chars()
            .filter(|&c| LanguageDetection::is_alpha(c))
            .count() as i64;
        if !named {
            if det.script != "latin" && det.script != "unknown" {
                if non_latin_words(sample).is_empty() || n_alpha < NON_LATIN_MIN_LETTERS {
                    continue;
                }
            } else if !(det.diacritic_rate >= NON_EN_DIACRITIC_RATE && word_runs(sample).len() >= 4)
            {
                continue;
            }
        }
        if n_alpha > best_n {
            best_n = n_alpha;
            best = Some(det);
        }
    }
    best
}
