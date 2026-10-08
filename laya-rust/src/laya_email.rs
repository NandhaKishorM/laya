//! Email utilities for cleaning and structuring email inputs. Ports `laya/email.py`.
//!
//! `_SENTENCE` (`(?<=[.!?])\s+`, a lookbehind) has no `regex`-crate equivalent — that crate has no
//! look-around support at all — so [`split_after_sentence`] is a small hand-written splitter
//! instead of pulling in a second regex engine (`fancy-regex`) for one pattern. Every other
//! pattern here (`_QUOTE_HEADERS`, `_SIGNATURE_MARKERS`, `_DISCLAIMER`, the paragraph splitter,
//! and the `[ \t]+` collapse) uses `regex`, whose Unicode support is on by default, so `(?i)`
//! case-folding and `\s` here are Unicode-aware the same way Python's `re.I` is.

use regex::Regex;
use serde_json::{Map, Value};
use std::sync::LazyLock;

/// `clean_email_body`'s default `max_chars`.
pub const DEFAULT_MAX_CHARS: usize = 3000;

static QUOTE_HEADERS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)^\s*On .{0,300}wrote:\s*$").unwrap(),
        Regex::new(r"(?i)^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}").unwrap(),
        Regex::new(r"^\s*_{8,}\s*$").unwrap(),
        Regex::new(r"(?i)^\s*From:\s.+$").unwrap(),
    ]
});

static SIGNATURE_MARKERS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"^\s*--\s*$").unwrap(),
        Regex::new(
            r"(?i)^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$",
        )
        .unwrap(),
        Regex::new(r"(?i)^\s*sent from my (iphone|android|mobile|ipad)").unwrap(),
    ]
});

static DISCLAIMER: LazyLock<Regex> = LazyLock::new(|| {
    // Two adjacent raw string literals (not one string with a trailing backslash-newline): a raw
    // string does not process escapes, so a `\` immediately before a real line break would stay
    // in the compiled pattern as a literal backslash followed by a newline, not a continuation.
    Regex::new(concat!(
        r"(?i)(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|",
        r"if you (have )?received this (e-?mail|message) in error)",
    ))
    .unwrap()
});

static PARAGRAPH_BREAK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\s*\n").unwrap());
static SPACE_TAB_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").unwrap());

/// Namespace for email cleaning and state construction. See the module docs for the source this
/// ports.
pub struct LayaEmail;

impl LayaEmail {
    /// Remove quoted email history, signatures and disclaimers to keep input focused. Ports
    /// `clean_email_body`.
    pub fn clean_body(body: &str, max_chars: usize) -> String {
        let text = body
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            // A literal two-character backslash-n (as opposed to an actual newline byte), which
            // some upstream systems send instead of a real line break.
            .replace("\\n", "\n");

        let mut lines: Vec<String> = Vec::new();
        for line in text.split('\n') {
            // The break needs `lines` (what has been kept so far) to be non-empty: a quote header
            // as the very first line is ordinary body text, not a reply separator, and is kept.
            if !lines.is_empty() && QUOTE_HEADERS.iter().any(|re| re.is_match(line)) {
                break;
            }
            if line.trim_start().starts_with('>') {
                continue;
            }
            lines.push(line.trim_end().to_string());
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
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        let joined_paragraphs = kept.join("\n\n");
        let collapsed = SPACE_TAB_RUN.replace_all(&joined_paragraphs, " ");
        collapsed.chars().take(max_chars).collect()
    }

    /// Construct a clean state object for email classification. Ports `email_state`.
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
        let mut map = Map::new();
        map.insert(
            "subject".to_string(),
            Value::String(subject.trim().to_string()),
        );
        let cleaned_body = if clean {
            Self::clean_body(body, DEFAULT_MAX_CHARS)
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
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .flat_map(|part| {
                if !DISCLAIMER.is_match(&part) || !part.contains('\n') {
                    return vec![part];
                }
                let mut pieces = Vec::new();
                let mut buf = String::new();
                for line in part.lines().map(str::trim).filter(|l| !l.is_empty()) {
                    let starts_sentence = line.chars().find(|&c| {
                        crate::language_detection::LanguageDetection::is_alpha(c)
                    }).is_some_and(char::is_uppercase);
                    if !buf.is_empty() && starts_sentence {
                        pieces.push(std::mem::take(&mut buf));
                    }
                    if !buf.is_empty() {
                        buf.push(' ');
                    }
                    buf.push_str(line);
                }
                if !buf.is_empty() {
                    pieces.push(buf);
                }
                pieces
            })
            .collect();
        parts
            .into_iter()
            .filter(|p| !DISCLAIMER.is_match(p))
            .collect::<Vec<_>>()
            .join(" ")
    }
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
        let trimmed = line.trim();
        if trimmed.chars().count() <= 40 && SIGNATURE_MARKERS.iter().any(|re| re.is_match(line)) {
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
        if i > 0 && chars[i].is_whitespace() && matches!(chars[i - 1], '.' | '!' | '?') {
            result.push(chars[start..i].iter().collect::<String>());
            while i < chars.len() && chars[i].is_whitespace() {
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
