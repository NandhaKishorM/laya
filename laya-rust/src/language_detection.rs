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
//! character, but not a plain decimal digit).

use std::collections::HashMap;
use unicode_general_category::{GeneralCategory, get_general_category};

/// Namespace for language/script detection. See the module docs for the source this ports.
pub struct LanguageDetection;

/// The full detection result for one piece of state. See [`LanguageDetection::analyse`].
#[derive(Debug, Clone, PartialEq)]
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
);

/// Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
/// weighted and a margin is required before calling something non-English.
///
/// The Romance lists (fr/es/pt/it) deliberately carry the *unaccented* function words as well as
/// the accented ones. A state that lost its accents -- mail clients, ticket systems and any
/// pipeline that normalises to ASCII strip them -- keeps no diacritic rate for the non-English
/// signal to read, so `la`, `un`, `y`, `e`, `et`, `deux` and friends are the only evidence left.
/// With a list of mostly accented words such a state produced one hit or none, fell under the
/// two-hit margin below, and was handed to the English checkpoint as undecided-but-not-non-English
/// text.
///
/// Order is load-bearing for `latin_profile`'s tie-break (`_STOP` iteration order: en, fr, de, es,
/// pt, it, nl, ro), so this stays a `Vec` of `(code, words)` pairs rather than a `HashMap`.
fn stop_lists() -> &'static [&'static str] {
    // Order is load-bearing (see the doc comment above): en, fr, de, es, pt, it, nl, ro.
    &["en", "fr", "de", "es", "pt", "it", "nl", "ro"]
}

// Slices (not fixed-size arrays) throughout, so splitting a long list across several `const`s
// for readability never needs a manually counted length.
const EN_STOP: &[&str] = &[
    "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that", "this",
    "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from", "will", "can",
    "would", "there", "their", "what", "which", "please", "we", "i",
];

const FR_STOP: &[&str] = &[
    "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur", "pas",
    "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce", "et", "du", "au",
    "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton", "ma", "ta", "sa", "mes", "tes",
    "ses", "ces", "deux", "trois", "très", "bien", "tout", "tous", "toute", "fait", "veux", "veut",
    "peux", "peut", "dois", "doit", "merci", "bonjour", "jour", "jours", "mois", "fois", "quand",
    "comment", "pourquoi", "alors", "donc",
];

const DE_STOP: &[&str] = &[
    "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für", "auf",
    "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
];

const ES_STOP: &[&str] = &[
    "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como", "pero",
    "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus", "la", "un", "y", "al", "lo",
    "le", "les", "su", "mi", "tu", "nos", "ni", "dos", "tres", "fue", "fueron", "ser", "tiene",
    "tienen", "tengo", "puede", "pueden", "quiero", "necesito", "hemos", "han", "sobre", "entre",
    "cuando", "donde", "porque", "aunque", "también", "ya", "eso", "esto", "esa", "ese", "nada",
    "algo", "aquí", "hoy", "gracias",
];

const PT_STOP: &[&str] = &[
    "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da", "dos",
    "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela", "o", "e", "na", "nas",
    "nos", "ao", "aos", "por", "foi", "era", "ser", "sou", "tem", "tenho", "pode", "podem",
    "quero", "preciso", "eu", "meu", "minha", "seu", "sua", "isso", "isto", "aqui", "ali", "como",
    "quando", "onde", "porque", "mais", "já", "ainda", "agora", "hoje", "ontem", "dois", "três",
    "tudo", "nada", "obrigado", "olá",
];

const IT_STOP: &[&str] = &[
    "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
    "questo", "questa", "anche", "come", "più", "nella", "alla", "la", "le", "un", "uno", "una",
    "e", "ed", "o", "da", "su", "tra", "fra", "mi", "ci", "ne", "ho", "hai", "ha", "abbiamo",
    "avete", "hanno", "era", "stato", "stata", "devo", "deve", "devono", "voglio", "vorrei", "mio",
    "mia", "tuo", "sua", "quando", "dove", "perche", "molto", "poco", "sempre", "mai", "già",
    "ancora", "adesso", "oggi", "ieri", "grazie", "ciao", "scusa", "nel", "nell", "negli", "sul",
    "sulla", "sulle", "dal", "dalla", "dallo", "dagli", "dei", "delle", "dello", "degli", "agli",
    "alle", "col",
];

const NL_STOP: &[&str] = &[
    "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan", "door",
    "maar", "ook", "worden", "deze", "naar", "wordt",
];

const RO_STOP: &[&str] = &[
    "și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără", "ale",
    "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această", "acesta",
    "aceasta", "mi", "ți", "vă", "nu",
];

fn words_for(code: &str) -> &'static [&'static str] {
    match code {
        "en" => EN_STOP,
        "fr" => FR_STOP,
        "de" => DE_STOP,
        "es" => ES_STOP,
        "pt" => PT_STOP,
        "it" => IT_STOP,
        "nl" => NL_STOP,
        "ro" => RO_STOP,
        other => unreachable!("no stop list for {other:?}"),
    }
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

    /// Whether `ch` matches Python's `[^\W\d_]` character class: a `\w` character (letter, `Nl`,
    /// `No` or `_`) that is neither a decimal digit (`\d` = category `Nd`) nor `_`. In practice
    /// this is "a letter, or a numeral that is not a plain decimal digit".
    fn is_word_not_digit_or_underscore(ch: char) -> bool {
        if ch == '_' {
            return false;
        }
        matches!(
            get_general_category(ch),
            GeneralCategory::UppercaseLetter
                | GeneralCategory::LowercaseLetter
                | GeneralCategory::TitlecaseLetter
                | GeneralCategory::ModifierLetter
                | GeneralCategory::OtherLetter
                | GeneralCategory::LetterNumber
                | GeneralCategory::OtherNumber
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

    /// Whether `cp` falls in the Latin block (Basic Latin through Latin-1 Supplement and the
    /// Latin Extended blocks below Latin Extended Additional, plus Latin Extended Additional
    /// itself) that `detect_script`/`script_profile` treat as "latin".
    fn is_latin_codepoint(cp: u32) -> bool {
        cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp)
    }

    /// Collect the string leaves of a state (str / object / array), so detection sees real
    /// content. Depths beyond 6 are dropped, and keys are ignored (they are usually English).
    fn iter_text(state: &serde_json::Value, depth: u32, out: &mut Vec<String>) {
        if depth > 6 {
            return;
        }
        match state {
            serde_json::Value::String(s) => out.push(s.clone()),
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

    /// Flatten a state into the text used for detection (keys are ignored: they are usually
    /// English). `max_chars` truncates by Unicode scalar value, matching Python's `[:max_chars]`
    /// string slice.
    pub fn state_text(state: &serde_json::Value, max_chars: usize) -> String {
        let mut parts = Vec::new();
        Self::iter_text(state, 0, &mut parts);
        parts.join(" ").chars().take(max_chars).collect()
    }

    /// [`Self::state_text`] with Python's default `max_chars=4000`.
    pub fn state_text_default(state: &serde_json::Value) -> String {
        Self::state_text(state, 4000)
    }

    /// Dominant script of `text`: `"latin"`, `"han"`, `"devanagari"`, ... or `"unknown"` if there
    /// are no letters.
    ///
    /// Ties are broken by first-inserted order: the non-Latin scripts in the order they first
    /// appear in `text`, then `"latin"` last (it is only ever added to the tally after the scan,
    /// so it never wins a tie against a non-Latin script present in the same text).
    pub fn detect_script(text: &str) -> String {
        let mut counts: Vec<(&'static str, i64)> = Vec::new();
        let mut latin: i64 = 0;
        for ch in text.chars() {
            if !Self::is_alpha(ch) {
                continue;
            }
            let cp = ch as u32;
            if Self::is_latin_codepoint(cp) {
                latin += 1;
                continue;
            }
            let name = Self::script_for_codepoint(cp).unwrap_or("other");
            match counts.iter_mut().find(|(n, _)| *n == name) {
                Some(entry) => entry.1 += 1,
                None => counts.push((name, 1)),
            }
        }
        counts.push(("latin", latin));

        let total: i64 = counts.iter().map(|(_, c)| c).sum();
        if total == 0 {
            return "unknown".to_string();
        }
        let mut best = 0usize;
        for i in 1..counts.len() {
            if counts[i].1 > counts[best].1 {
                best = i;
            }
        }
        counts[best].0.to_string()
    }

    /// Fraction of alphabetic characters belonging to each detected script.
    pub fn script_profile(text: &str) -> HashMap<String, f64> {
        let mut counts: HashMap<&'static str, i64> = HashMap::new();
        counts.insert("latin", 0);
        for ch in text.chars() {
            if !Self::is_alpha(ch) {
                continue;
            }
            let cp = ch as u32;
            if Self::is_latin_codepoint(cp) {
                *counts.get_mut("latin").unwrap() += 1;
                continue;
            }
            let name = Self::script_for_codepoint(cp).unwrap_or("other");
            *counts.entry(name).or_insert(0) += 1;
        }
        let total: i64 = counts.values().sum();
        if total == 0 {
            return HashMap::new();
        }
        counts
            .into_iter()
            .filter(|&(_, v)| v != 0)
            .map(|(k, v)| (k.to_string(), v as f64 / total as f64))
            .collect()
    }

    /// Words in `text`, split on runs of Python's `[^\W\d_]` character class and lowercased with
    /// full Unicode case mapping.
    fn words(text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if Self::is_word_not_digit_or_underscore(ch) {
                current.push(ch);
            } else if !current.is_empty() {
                words.push(std::mem::take(&mut current).to_lowercase());
            }
        }
        if !current.is_empty() {
            words.push(current.to_lowercase());
        }
        words
    }

    /// Evidence behind the Latin-script language guess: `language` (may be `None` when
    /// undecided), `english_hits`, `diacritic_rate` and `looks_non_english`. [`Self::analyse`]
    /// needs the evidence and not just the verdict, because "undecided" and "English" are
    /// different answers and only one of them is safe to send to the English checkpoint.
    ///
    /// A non-English language is only named when it matched at least one word that no other list
    /// claims: shared function words alone (`la`, `e`, `o`) identify no particular language.
    pub fn latin_profile(text: &str) -> LatinProfile {
        let words = Self::words(text);
        let lowered = text.to_lowercase();
        let diac = lowered
            .chars()
            .filter(|c| NON_EN_DIACRITICS.contains(*c))
            .count();
        let diac_rate = diac as f64 / (lowered.chars().count().max(1) as f64);
        let non_english = diac_rate >= NON_EN_DIACRITIC_RATE;

        if words.len() < 4 {
            return LatinProfile {
                language: None,
                english_hits: 0,
                diacritic_rate: diac_rate,
                looks_non_english: non_english,
            };
        }

        let word_set: std::collections::HashSet<&str> = words.iter().map(String::as_str).collect();

        let lists = stop_lists();
        // Word counted more than once across the stop lists: a word may not name a language by
        // itself, though it still counts toward that language's score.
        let mut membership: HashMap<&str, i32> = HashMap::new();
        for &code in lists {
            for w in words_for(code) {
                *membership.entry(w).or_insert(0) += 1;
            }
        }
        let shared_words: std::collections::HashSet<&str> = membership
            .into_iter()
            .filter(|&(_, count)| count > 1)
            .map(|(w, _)| w)
            .collect();

        let mut scores: HashMap<&str, i64> = HashMap::new();
        for &code in lists {
            let set: std::collections::HashSet<&str> = words_for(code).iter().copied().collect();
            let score = words.iter().filter(|w| set.contains(w.as_str())).count() as i64;
            scores.insert(code, score);
        }
        let en = *scores.get("en").unwrap_or(&0);

        // Only a language that matched at least one word no other list claims may be named.
        // Evaluated (and the winner picked) in `stop_lists()` order, so ties keep the earlier
        // language, matching Python's dict iteration + `max()`.
        let mut best_lg: Option<&str> = None;
        let mut best: i64 = 0;
        for &code in lists {
            if code == "en" {
                continue;
            }
            let list_set: std::collections::HashSet<&str> =
                words_for(code).iter().copied().collect();
            let has_own_word = word_set
                .intersection(&list_set)
                .any(|w| !shared_words.contains(w));
            if !has_own_word {
                continue;
            }
            let score = scores[code];
            if best_lg.is_none() || score > best {
                best_lg = Some(code);
                best = score;
            }
        }

        let mut language = None;
        if let Some(lg) = best_lg {
            if best >= 2i64.max(en + 2) {
                // A non-English language needs a clear margin over English function words.
                language = Some(lg.to_string());
            } else if non_english && best >= 2i64.max(en) {
                // Needs two hits here too: one shared function word ("para" in Turkish text)
                // named Spanish on the strength of the diacritics alone, which is a guess dressed
                // as a detection.
                language = Some(lg.to_string());
            }
        }
        if language.is_none() && en > 0 && !non_english {
            language = Some("en".to_string());
        }

        LatinProfile {
            language,
            english_hits: en,
            diacritic_rate: diac_rate,
            looks_non_english: non_english,
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

    /// Non-Latin words, excluding one-letter symbols and capitalised proper names.
    fn has_non_latin_words(text: &str) -> bool {
        let mut current = String::new();
        let mut script = None;
        let is_word = |word: &str| {
            word.chars().count() >= 2
                && word.chars().next().is_some_and(|c| !c.is_uppercase())
        };
        for ch in text.chars() {
            if matches!(get_general_category(ch), GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark | GeneralCategory::EnclosingMark)
            {
                continue;
            }
            let next = if Self::is_alpha(ch) && !Self::is_latin_codepoint(ch as u32) {
                Self::script_for_codepoint(ch as u32)
            } else {
                None
            };
            if next.is_some() && next == script {
                current.push(ch);
                continue;
            }
            if is_word(&current) {
                return true;
            }
            current.clear();
            script = next;
            if next.is_some() {
                current.push(ch);
            }
        }
        is_word(&current)
    }

    /// Full detection result for a state. See [`LanguageAnalysis`].
    pub fn analyse(state: &serde_json::Value) -> LanguageAnalysis {
        let text = Self::state_text_default(state);
        let prof = Self::script_profile(&text);
        let mut script = Self::detect_script(&text);
        let non_latin = if prof.is_empty() {
            0.0
        } else {
            crate::calibration::Calibration::round4(1.0 - prof.get("latin").copied().unwrap_or(0.0))
        };
        let n_non_latin = (non_latin * text.chars().filter(|&c| Self::is_alpha(c)).count() as f64)
            .round_ties_even();
        if script == "latin" && Self::has_non_latin_words(&text)
            && (non_latin >= 0.2 || (non_latin >= 0.1 && n_non_latin >= 10.0))
        {
            // Preserve Python's first-seen tie order, rather than HashMap iteration order.
            let mut best = 0.0;
            for ch in text.chars().filter(|&c| Self::is_alpha(c)) {
                if Self::is_latin_codepoint(ch as u32) {
                    continue;
                }
                let name = Self::script_for_codepoint(ch as u32).unwrap_or("other");
                let fraction = prof.get(name).copied().unwrap_or(0.0);
                if fraction > best {
                    script = name.to_string();
                    best = fraction;
                }
            }
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
            };
        }
        if script != "latin" {
            return LanguageAnalysis {
                script,
                script_profile: prof,
                language: None,
                is_english: false,
                language_undecided: true,
                diacritic_rate: 0.0,
                non_latin_fraction: non_latin,
            };
        }

        let prof_lat = Self::latin_profile(&text);
        let lang = prof_lat.language.clone();
        // Undecided is not English. Treating it as English sent every Latin-script language with
        // no stopword list here to the checkpoint that cannot read it, silently. When nothing
        // identifies the language, non-English letters are enough to prefer the multilingual
        // checkpoint; text with no such letters (including short English) still goes to the
        // English one.
        let undecided = lang.is_none();
        let english = lang.as_deref() == Some("en") || (undecided && !prof_lat.looks_non_english);

        LanguageAnalysis {
            script: "latin".to_string(),
            script_profile: prof,
            language: lang,
            is_english: english,
            language_undecided: undecided,
            diacritic_rate: crate::calibration::Calibration::round4(prof_lat.diacritic_rate),
            non_latin_fraction: non_latin,
        }
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
    /// Whether the diacritic rate alone is high enough to call the text non-English.
    pub looks_non_english: bool,
}
