//! Email utilities for cleaning and structuring email inputs. Ports `laya/email.py`.
//!
//! The markers cover English, Portuguese, Spanish and French mail clients: Gmail's `Em ...
//! escreveu:` / `Le ... a écrit :`, Outlook's `-----Mensagem original-----`, the `Atenciosamente`
//! and `Cordialement` sign-offs, `Envoyé depuis mon iPhone` and the confidentiality footers.
//! With English-only markers cleaning was a no-op for those mails and the quoted history (often
//! a *different* request) weighed on the answer as much as the new message did.
//!
//! The `regex` crate has no look-around, so the three lookaheads of `email.py` (`(?=.*\d)` on
//! the dated attributions) are a separate "the line holds a digit" test next to a pattern without
//! them, and `_SENTENCE` (`(?<=[.!?])\s+`, a lookbehind) is the small hand-written
//! [`split_after_sentence`]. The English sign-off (`_is_english_signoff`) is hand-written too:
//! its name rule is a Unicode category test on each token's first letter. Every other pattern
//! uses `regex`, behind [`PyRegex`], which rewrites the haystack so the crate's `\w`/`\b` and
//! `(?i)` agree with Python's `re` (see its docs); `\s` is widened by [`py_regex`] to cover the
//! U+001C..U+001F that Python's `\s` accepts and the crate's does not.

use crate::language_detection::{
    LanguageDetection, char_is_upper, is_letter_word, is_word_char, prefix_chars, py_is_space,
    py_strip,
};
use regex::{Regex, RegexBuilder};
use serde_json::{Map, Value};
use std::borrow::Cow;
use std::sync::LazyLock;
use unicode_general_category::{GeneralCategory, get_general_category};

/// `clean_email_body`'s default `max_chars`.
pub const DEFAULT_MAX_CHARS: usize = 3000;

/// Compile a pattern the way Python's `re` reads it: `\s` / `\S` also cover U+001C..U+001F. The
/// size limit is raised because the disclaimer pattern repeats Unicode classes (`[^.]{0,80}`).
fn py_regex(pattern: &str) -> Regex {
    let widened = pattern
        .replace(r"\s", r"[\s\x1c-\x1f]")
        .replace(r"\S", r"[^\s\x1c-\x1f]");
    RegexBuilder::new(&widened)
        .size_limit(256 * 1024 * 1024)
        .build()
        .expect("email pattern is valid")
}

/// A pattern run the way Python's `re` runs it. Two things differ between `re` and the `regex`
/// crate for the patterns here, and neither can be switched off in the crate:
///
/// - `\w` (so `\b`): Python's is `str.isalnum()` or `_`, so a superscript or other `No` digit
///   is a word character, and a combining mark, a joiner, a connector other than `_` or a
///   circled letter is not. The crate's `\w` is the other way round on exactly those.
/// - `re.I`: Python also folds U+0130 and U+0131 to `i` (its simple lower-casing maps the first
///   to `i`, and `_casefix` adds the second); the crate's simple case folding does not.
///
/// So the haystack is rewritten one character for one character before it reaches the crate:
/// each character whose word-ness the two disagree on becomes a placeholder with Python's
/// word-ness, and U+0130/U+0131 become `i`. The placeholders occur in no pattern and are neither
/// whitespace nor digits, so nothing else about a match changes. Pure ASCII text is identical
/// in both engines and goes through untouched.
struct PyRegex {
    regex: Regex,
}

/// A letter that no pattern here mentions: stands in for a character Python treats as `\w` and
/// the crate does not.
const PY_WORD_ONLY: char = '\u{416}';
/// A non-word, non-space character that no pattern here mentions: stands in for a character the
/// crate treats as `\w` and Python does not.
const REGEX_WORD_ONLY: char = '#';

static REGEX_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\w$").unwrap());

fn as_python_sees_it(ch: char) -> char {
    if ch.is_ascii() {
        return ch;
    }
    if ch == '\u{130}' || ch == '\u{131}' {
        return 'i';
    }
    let mut buf = [0u8; 4];
    let regex_word = REGEX_WORD.is_match(ch.encode_utf8(&mut buf));
    match (is_word_char(ch), regex_word) {
        (true, false) => PY_WORD_ONLY,
        (false, true) => REGEX_WORD_ONLY,
        _ => ch,
    }
}

impl PyRegex {
    fn new(pattern: &str) -> Self {
        Self {
            regex: py_regex(pattern),
        }
    }

    fn view(text: &str) -> Cow<'_, str> {
        if text.is_ascii() {
            Cow::Borrowed(text)
        } else {
            Cow::Owned(text.chars().map(as_python_sees_it).collect())
        }
    }

    /// `pattern.search(text)` is not `None`.
    fn is_match(&self, text: &str) -> bool {
        self.regex.is_match(&Self::view(text))
    }

    /// The byte offset in `text` where the leftmost match of `pattern.search(text)` ends.
    fn find_end(&self, text: &str) -> Option<usize> {
        let view = Self::view(text);
        let end = self.regex.find(&view)?.end();
        // The rewrite is one character for one character, so count characters, not bytes.
        let chars = view[..end].chars().count();
        Some(
            text.char_indices()
                .nth(chars)
                .map_or(text.len(), |(offset, _)| offset),
        )
    }
}

/// A quote-header pattern, plus the rule `(?=.*\d)` that `regex` cannot express: the dated
/// attribution of a mail client always carries a date, so a line without a digit is body text
/// ("Em resposta ao que você escreveu:").
struct QuoteHeader {
    pattern: PyRegex,
    needs_digit: bool,
}

impl QuoteHeader {
    fn new(pattern: &str, needs_digit: bool) -> Self {
        Self {
            pattern: PyRegex::new(pattern),
            needs_digit,
        }
    }

    fn matches(&self, line: &str) -> bool {
        self.pattern.is_match(line) && (!self.needs_digit || has_decimal_digit(line))
    }
}

/// Python's `\d`: a decimal digit (`Nd`) in any script.
fn is_decimal(ch: char) -> bool {
    get_general_category(ch) == GeneralCategory::DecimalNumber
}

fn has_decimal_digit(line: &str) -> bool {
    line.chars().any(is_decimal)
}

static QUOTE_HEADERS: LazyLock<Vec<QuoteHeader>> = LazyLock::new(|| {
    vec![
        QuoteHeader::new(r"(?i)^\s*On .{0,300}wrote:\s*$", false),
        // "Em resposta ao que você escreveu:" is body text; a client's attribution always
        // carries a date.
        QuoteHeader::new(r"(?i)^\s*Em .{0,300}escreveu:\s*$", true),
        QuoteHeader::new(r"(?i)^\s*El .{0,300}escribi[óo]:\s*$", true),
        // Gmail's French attribution always carries a date too ("Le lun. ... a écrit :"); a
        // bare "Le rapport que vous avez écrit :" is body text. French typography puts a space
        // before the colon, so the verb allows one (`a écrit :`, never `a écrit:`).
        QuoteHeader::new(r"(?i)^\s*Le .{0,300}a [eé]crit\s*:\s*$", true),
        QuoteHeader::new(
            r"(?i)^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}",
            false,
        ),
        QuoteHeader::new(
            concat!(
                r"(?i)^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado)|",
                r"Message d'origine)\s*-{2,}",
            ),
            false,
        ),
        QuoteHeader::new(r"^\s*_{8,}\s*$", false),
        // `From:` opens ordinary prose too ("From: my side the integration works, but please
        // refund..."), and a reply header always carries the sender, so the header is only
        // recognised when an address follows -- the same rule as `De:` below. A bare
        // `From: Name` header is caught by HEADER_FROM_NAME/HEADER_NEXT instead, which need the
        // header's own `Sent:`/`Date:` line to tell it apart from a sentence.
        QuoteHeader::new(r"(?i)^\s*From:\s.*[@<]", false),
        // `De:` also opens ordinary Portuguese/Spanish lines ("De: 10/09 a 15/09"), so the
        // Outlook header is only recognised when it carries an address. French Outlook writes
        // `De :` with a space, so the marker allows one; the address rule is unchanged.
        QuoteHeader::new(r"(?i)^\s*De\s*:\s.*[@<]", false),
    ]
});

/// Gmail wraps a long attribution line, leaving `fulano@x.com> escreveu:` alone on the next
/// line. That tail cuts too, and takes the `On/Em/El/Le ...` head it belongs to with it. The
/// French tail keeps its spaced colon (`support@x.com> a écrit :`).
static ATTRIBUTION_TAIL: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(r"(?i)^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]|a [eé]crit)\s*:\s*$")
});
/// The head the tail belongs to: `On|Em|El|Le`, then a date (`(?=.*\d)`, checked separately).
static ATTRIBUTION_HEAD: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(r"(?i)^\s*(On|Em|El|Le) "));

/// Exchange often leaves the address out of Outlook's reply header ("De: Maria Souza"), so a
/// bare `De:` only cuts when the header's own `Enviado:` line, or a dated `Data:`/`Fecha:` line,
/// follows it. `Para:` is not enough: "De: 10/09 / Para: 15/09" is how a leave request reads. The
/// same is true of a bare English `From: Maria Souza`; French Outlook writes `De : Marie Dupont`
/// with `Envoyé :` underneath. Both allow the French spaced colon; the neighbours still do the
/// telling apart.
static HEADER_FROM_NAME: LazyLock<PyRegex> =
    LazyLock::new(|| PyRegex::new(r"(?i)^\s*(De|From)\s*:\s+\S"));
static HEADER_NEXT: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(concat!(
        r"(?i)^\s*(Enviad[oa]( em| el)?:\s|Envoy[ée]( le)?\s*:\s|Sent:\s|",
        r"(Data|Fecha|Date):\s.*\d{4})",
    ))
});

/// Closing words of an English sign-off, with the optional suffix a closing may carry.
static SIGNOFF_HEAD: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(concat!(
        r"^\s*(?i:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)",
        r"(?i:\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?",
    ))
});

static SIGNATURE_DASHES: LazyLock<PyRegex> = LazyLock::new(|| PyRegex::new(r"^\s*--\s*$"));

/// Portuguese/Spanish/French sign-offs match only on their own: "Obrigado pelo retorno, mas ..."
/// is a request, not a signature, so unlike the English marker no trailing words are allowed.
/// "Merci pour votre aide, mais ..." stays, while a bare "Cordialement," or "Merci," goes with
/// the name underneath it.
static SIGNOFF_TRANSLATED: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(concat!(
        r"(?i)^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|",
        r"(muito )?obrigad[oa]s?( desde já| pela atenção)?|(com os melhores )?cumprimentos|",
        r"saudações|(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?|",
        r"(bien )?cordialement|salutations( distinguées)?|bien à vous|merci( d'avance)?|",
        r"bonne journée)[\s,!.]*$",
    ))
});

/// Mobile and mail-app footers. Only a line that is nothing *but* the footer matches -- "Enviado
/// do meu celular o comprovante ontem." is a request -- and such a line may run to 60
/// characters, since Samsung's default ("Enviado do meu smartphone Samsung Galaxy.") is longer
/// than a sign-off's 40. Apple's French default ("Envoyé depuis mon iPhone") reads the same way.
const DEVICE: &str = concat!(
    "iphone|ipad|android|ios|mobile|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|",
    "outlook|yahoo|mail|e-?mail|gmail|windows",
);
static DEVICE_FOOTER: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(&format!(
        concat!(
            r"(?i)^\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|",
            r"sent from( my)?|envoy[ée] (depuis|de)( mon| ma| mes)?)",
            r" ({d})( ({d}|para|for|no|na|\d+|phone|device|pro|max|mini|plus|using [a-z][a-z0-9_.+-]*))*",
            r"|(obter o|get) outlook (para|for) (ios|android))[\s.!]*$",
        ),
        d = DEVICE
    ))
});

static DISCLAIMER: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(concat!(
        // English: tied to a disclaimer noun and a disclaimer tail, the way the Portuguese
        // branches below are. The bare word matched any sentence that merely mentioned it, so
        // "Is this confidential?" and "Confidential: I need a refund." were deleted whole.
        // `[^.]` rather than `[^.\n]`: a footer wraps, so "are\nconfidential" must still match.
        r"(?i)(\b(e-?mail|message|information|communication|transmission|contents?)\b[^.]{0,60}",
        r"\bconfidential\b[^.]{0,60}\b(intended|solely|addressee|recipient|privileged|",
        r"disclos|unauthori[sz]ed)|",
        r"\bconfidential\b[^.]{0,60}\b(and (may|is) (also )?privileged)|",
        r"if you (have )?received this (e-?mail|message) in error|",
        // Portuguese/Spanish: tied to "this message/e-mail" rather than the bare word
        // `confidencial`, which a sender's own request ("preciso do contrato confidencial")
        // uses just as often.
        r"\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|",
        r"\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}",
        r"(destinatári|destinatari|pessoa|persona|entidade|entidad)|",
        r"\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|",
        r"\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error|",
        // the "think before printing" footer, tied to its environmental ending rather than to
        // `antes de imprimir`, which a request uses too ("antes de imprimir o boleto, confira
        // o valor")
        r"\bantes de imprimir\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|",
        r"\b(meio|medio) ambiente\b[^.]{0,30}antes de imprimir|",
        // French: tied to "ce message/cet e-mail" rather than the bare word `confidentiel`,
        // which a sender's own request ("le contrat confidentiel") uses just as often.
        r"\b(ce|cet|cette) (message|e-?mail|mail|courriel)\b[^.]{0,80}(confidentiel|privil[eé]gi)|",
        r"\bavez re[çc]u (ce|cet|cette) (message|e-?mail|mail)\b[^.]{0,20} par erreur|",
        r"\b(usage exclusif|exclusivement|uniquement)\b[^.]{0,30}destinataire)",
    ))
});

static PARAGRAPH_BREAK: LazyLock<Regex> = LazyLock::new(|| py_regex(r"\n\s*\n"));
static SPACE_TAB_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").unwrap());

/// `_drop_marks`: remove combining marks (`Mn`/`Mc`/`Me`, which the TS port spells `\p{M}`) that
/// ride on a preceding character. A mark that opens the string or follows a space has no base to
/// ride on and stays, so it still separates tokens.
fn drop_marks(text: &str) -> Vec<char> {
    let mut kept: Vec<char> = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let is_mark = matches!(
            get_general_category(ch),
            GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        );
        if is_mark && kept.last().is_some_and(|&last| !py_is_space(last)) {
            continue;
        }
        kept.push(ch);
    }
    kept
}

/// `_SIGNOFF_TAIL`: `^[\s,;:!.]*(?:[^\W\d_][\w'-]*[\s,.]*){0,3}$` -- punctuation, then at most
/// three name tokens. A token is a letter, then word characters, apostrophes and hyphens, then
/// separators; splitting a word differently can only add tokens, so the word-level reading is
/// the one that decides.
fn signoff_tail_matches(tail: &[char]) -> bool {
    let n = tail.len();
    let mut i = 0;
    while i < n && (py_is_space(tail[i]) || matches!(tail[i], ',' | ';' | ':' | '!' | '.')) {
        i += 1;
    }
    let mut tokens = 0;
    while i < n {
        if !is_letter_word(tail[i]) {
            return false;
        }
        tokens += 1;
        if tokens > 3 {
            return false;
        }
        i += 1;
        while i < n && (is_word_char(tail[i]) || matches!(tail[i], '\'' | '-')) {
            i += 1;
        }
        while i < n && (py_is_space(tail[i]) || matches!(tail[i], ',' | '.')) {
            i += 1;
        }
    }
    true
}

/// `_is_english_signoff`: a closing word followed by nothing but punctuation and a short name.
///
/// A closing line is the closing word plus punctuation and at most a name. Anything else on the
/// line is a sentence, and the case of the next word is what separates the two: a name is
/// capitalised, "for" in "Thanks for the quick reply." is not. A name starts with a letter that
/// is not lowercase -- capitalised in any script (`Łukasz`, `Дмитрий`) or caseless (`山田`) -- so
/// each token's first letter is judged by category (`Lu`, `Lt`, `Lo`). Combining marks ride along
/// with the letter before them (`Jose\u0301` is `José`).
fn is_english_signoff(line: &str) -> bool {
    let Some(head_end) = SIGNOFF_HEAD.find_end(line) else {
        return false;
    };
    let tail = drop_marks(&line[head_end..]);
    if !signoff_tail_matches(&tail) {
        return false;
    }
    // `_SIGNOFF_TOKEN.findall(tail)`: `[^\W\d_][\w'-]*`
    let mut i = 0;
    while i < tail.len() {
        if is_letter_word(tail[i]) {
            if !matches!(
                get_general_category(tail[i]),
                GeneralCategory::UppercaseLetter
                    | GeneralCategory::TitlecaseLetter
                    | GeneralCategory::OtherLetter
            ) {
                return false;
            }
            i += 1;
            while i < tail.len() && (is_word_char(tail[i]) || matches!(tail[i], '\'' | '-')) {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    true
}

/// Whether `line` is a signature marker: `--`, an English sign-off, or a Portuguese/Spanish/
/// French one.
fn is_signature_marker(line: &str) -> bool {
    SIGNATURE_DASHES.is_match(line) || is_english_signoff(line) || SIGNOFF_TRANSLATED.is_match(line)
}

/// Python's `s[:n]` for a possibly negative `n`: a negative `n` drops that many characters from
/// the end (everything, once it reaches the length).
fn py_prefix(s: &str, n: i64) -> String {
    if n >= 0 {
        return prefix_chars(s, usize::try_from(n).unwrap_or(usize::MAX)).to_string();
    }
    let drop = usize::try_from(n.unsigned_abs()).unwrap_or(usize::MAX);
    let len = s.chars().count();
    prefix_chars(s, len.saturating_sub(drop)).to_string()
}

/// Namespace for email cleaning and state construction. See the module docs for the source this
/// ports.
pub struct LayaEmail;

impl LayaEmail {
    /// Remove quoted email history, signatures and disclaimers to keep input focused. Ports
    /// `clean_email_body`; see [`Self::clean_body_signed`] for the budget arithmetic.
    pub fn clean_body(body: &str, max_chars: usize) -> String {
        Self::clean_body_signed(body, i64::try_from(max_chars).unwrap_or(i64::MAX))
    }

    /// [`Self::clean_body`] with Python's signed `max_chars`, whose edge cases are specified
    /// behaviour (the `budget-zero` and `budget-negative-*` cases every port is tested against).
    ///
    /// `max_chars` is a slice budget, not a floor, and it is applied twice: as the
    /// `max_chars * 4` pre-cut that bounds the regex work below, and as the `text[:max_chars]`
    /// that returns the result. The pre-cut runs first, so the quote, signature and disclaimer
    /// rules see only the first `4 * max_chars` characters -- a footer that begins past that
    /// window is never recognized, and the paragraph holding it survives to the final cut. So `0`
    /// answers with the empty string, and a negative value trims from the end: the pre-cut
    /// always fires and removes `4 * |max_chars|`, and the final slice removes `|max_chars|`
    /// more, which is `5 * |max_chars|` off the body and empty once that reaches the body's
    /// length. "No limit" means a budget larger than the body -- never `0` and never negative.
    pub fn clean_body_signed(body: &str, max_chars: i64) -> String {
        let text = body
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            // A literal two-character backslash-n (as opposed to an actual newline byte), which
            // some upstream systems send instead of a real line break.
            .replace("\\n", "\n");
        // Bound regex work before the expensive patterns below: the disclaimer pattern uses
        // `[^.]{0,60/80/100}` alternations whose cost grows with input length, and only
        // `max_chars` are ever returned.
        let pre_cut = max_chars.saturating_mul(4);
        let text = if i64::try_from(text.chars().count()).unwrap_or(i64::MAX) > pre_cut {
            py_prefix(&text, pre_cut)
        } else {
            text
        };

        let mut lines: Vec<String> = Vec::new();
        let source: Vec<&str> = text.split('\n').collect();
        for (i, line) in source.iter().enumerate() {
            // The breaks need `lines` (what has been kept so far) to be non-empty: a quote
            // header as the very first line is ordinary body text, not a reply separator.
            if !lines.is_empty() && QUOTE_HEADERS.iter().any(|header| header.matches(line)) {
                break;
            }
            if !lines.is_empty()
                && HEADER_FROM_NAME.is_match(line)
                && source
                    .get(i + 1)
                    .is_some_and(|next| HEADER_NEXT.is_match(next))
            {
                break;
            }
            if !lines.is_empty() && ATTRIBUTION_TAIL.is_match(line) {
                if lines
                    .last()
                    .is_some_and(|last| ATTRIBUTION_HEAD.is_match(last) && has_decimal_digit(last))
                {
                    lines.pop();
                }
                break;
            }
            if line.trim_start_matches(py_is_space).starts_with('>') {
                continue;
            }
            lines.push(line.trim_end_matches(py_is_space).to_string());
        }

        let cut = signature_cut(&lines);
        lines.truncate(cut);

        let joined = lines.join("\n");
        let paragraphs: Vec<String> = PARAGRAPH_BREAK
            .split(&joined)
            .map(Self::strip_disclaimer)
            .collect();
        let kept: Vec<&str> = paragraphs
            .iter()
            .map(|p| py_strip(p))
            .filter(|p| !p.is_empty())
            .collect();
        let joined_paragraphs = kept.join("\n\n");
        let collapsed = SPACE_TAB_RUN.replace_all(&joined_paragraphs, " ");
        py_prefix(&collapsed, max_chars)
    }

    /// Construct a clean state object for email classification. Ports `email_state` at its
    /// default `max_chars` (3000).
    ///
    /// `extra` mirrors Python's `**extra`: only entries whose value is not `Value::Null` are
    /// included, and they are added after `subject`/`body`/`from` in the order given.
    pub fn state(
        subject: &str,
        body: &str,
        sender: Option<&str>,
        clean: bool,
        extra: impl IntoIterator<Item = (String, Value)>,
    ) -> Value {
        Self::state_with_max_chars(
            subject,
            body,
            sender,
            clean,
            i64::try_from(DEFAULT_MAX_CHARS).unwrap_or(i64::MAX),
            extra,
        )
    }

    /// [`Self::state`] with Python's `max_chars`, the budget [`Self::clean_body_signed`] cuts the
    /// body to. It is worth raising for a long message: at the default the body stops after 3000
    /// characters, so a request in the last paragraphs never reaches the model -- including
    /// through `predict_long`. Lowering it does the same thing more surely: `0` gives the model a
    /// subject and an empty body, and a negative value cuts from the end, which is where that
    /// late request lives. Pass a budget larger than the body when the message is long, never
    /// `0` or a negative to mean "no limit". Ignored when `clean` is false, which passes the
    /// body through whole.
    pub fn state_with_max_chars(
        subject: &str,
        body: &str,
        sender: Option<&str>,
        clean: bool,
        max_chars: i64,
        extra: impl IntoIterator<Item = (String, Value)>,
    ) -> Value {
        let mut map = Map::new();
        map.insert(
            "subject".to_string(),
            Value::String(py_strip(subject).to_string()),
        );
        let cleaned_body = if clean {
            Self::clean_body_signed(body, max_chars)
        } else {
            body.to_string()
        };
        map.insert("body".to_string(), Value::String(cleaned_body));
        if let Some(sender) = sender
            && !sender.is_empty()
        {
            map.insert("from".to_string(), Value::String(sender.to_string()));
        }
        for (key, value) in extra {
            if !value.is_null() {
                map.insert(key, value);
            }
        }
        Value::Object(map)
    }

    /// Drop boilerplate disclaimer text from one paragraph.
    ///
    /// A paragraph is dropped whole only when *every* sentence in it is boilerplate; otherwise
    /// only the boilerplate sentences go. A footer that runs on without a blank line used to take
    /// the sender's actual request with it, which is worse than leaving one boilerplate line
    /// behind.
    fn strip_disclaimer(paragraph: &str) -> String {
        if !DISCLAIMER.is_match(paragraph) {
            return paragraph.to_string(); // nothing to do: keep the original line structure
        }
        let parts: Vec<String> = split_after_sentence(paragraph)
            .into_iter()
            .map(|p| py_strip(&p).to_string())
            .filter(|p| !p.is_empty())
            .flat_map(|part| {
                if DISCLAIMER.is_match(&part) {
                    split_fused_lines(&part)
                } else {
                    vec![part]
                }
            })
            .collect();
        parts
            .into_iter()
            .filter(|p| !DISCLAIMER.is_match(p))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// `_starts_new_sentence`: the first letter is uppercase, a fresh sentence rather than a wrapped
/// line. Lines in uncased scripts (CJK, Devanagari, ...) never start a new piece, so wrapped
/// boilerplate in those scripts still drops whole.
fn starts_new_sentence(line: &str) -> bool {
    line.chars()
        .find(|&c| LanguageDetection::is_alpha(c))
        .is_some_and(char_is_upper)
}

/// `_split_fused_lines`: split a fused boilerplate-positive sentence at sentence-starting
/// newlines. An unpunctuated request line glued to a disclaimer line ("locked\nThis ...") splits
/// at the newline because the next line starts uppercase; a lowercase continuation
/// ("are\nconfidential") belongs to the same sentence, so a wrapped footer still drops whole.
fn split_fused_lines(sentence: &str) -> Vec<String> {
    if !sentence.contains('\n') {
        return vec![sentence.to_string()];
    }
    let mut pieces = Vec::new();
    let mut buf = String::new();
    for line in sentence.split('\n').map(py_strip).filter(|l| !l.is_empty()) {
        if !buf.is_empty() && starts_new_sentence(line) {
            pieces.push(std::mem::take(&mut buf));
            buf = line.to_string();
        } else {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(line);
        }
    }
    if !buf.is_empty() {
        pieces.push(buf);
    }
    pieces
}

/// Where to cut `lines` for a trailing signature block. Ports the `for i in range(...)` scan in
/// `clean_email_body`: only the last ~40% of lines (but always at least the last 8, and never
/// before line 1) are searched, so a short, sign-off-shaped line near the top of a genuine body
/// is never mistaken for a signature.
fn signature_cut(lines: &[String]) -> usize {
    let len = lines.len() as i64;
    // Python: `int(len(lines) * 0.6)` floors (towards zero; `len` is never negative, so this is
    // an ordinary floor). `as i64` on a non-negative f64 truncates the same way.
    let scan_from = ((len as f64) * 0.6) as i64;
    let scan_from = scan_from.min(len - 8).max(1);
    let scan_from = scan_from.max(0) as usize;

    for (i, line) in lines.iter().enumerate().skip(scan_from) {
        let n = py_strip(line).chars().count();
        if (n <= 40 && is_signature_marker(line)) || (n <= 60 && DEVICE_FOOTER.is_match(line)) {
            return i;
        }
    }
    lines.len()
}

/// Hand-written equivalent of `re.split(r"(?<=[.!?])\s+", text)`: splits on a run of whitespace
/// that is immediately preceded by `.`, `!` or `?`. The punctuation itself stays with the
/// preceding piece (the lookbehind is zero-width, so it is never part of the match that gets
/// removed); only the whitespace run is consumed.
fn split_after_sentence(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut result = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        if i > 0 && py_is_space(chars[i]) && matches!(chars[i - 1], '.' | '!' | '?') {
            result.push(chars[start..i].iter().collect::<String>());
            while i < chars.len() && py_is_space(chars[i]) {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }
    result.push(chars[start..].iter().collect());
    result
}
