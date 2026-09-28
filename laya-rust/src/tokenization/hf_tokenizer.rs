//! [`LayaTokenizer`] backed by the `tokenizers` crate — the same HuggingFace Rust `tokenizers`
//! library the Python SDK uses.
//!
//! ## No `tokenizer.nopost.json` strip-and-cache step needed
//!
//! `tokenizer.json` ships with a `TemplateProcessing` post-processor that wraps every encode in
//! special tokens (`<bos> A <eos>` for multilingual, `[CLS] A [SEP]` for the ModernBERT
//! checkpoints). Python bypasses it by calling the tokenizer with `add_special_tokens=False`. The
//! `tokenizers` Rust crate used here exposes `Tokenizer::encode(text, add_special_tokens: false)`
//! directly — exactly what Python does — so there is nothing to strip, cache, or fingerprint.

use crate::error::{LayaError, Result};
use crate::tokenization::{LayaTokenizer, SpecialTokens};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer as InnerTokenizer;

/// [`LayaTokenizer`] backed by `tokenizer/tokenizer.json`. See the module docs for what this
/// ports.
pub struct HfTokenizer {
    inner: InnerTokenizer,
    special: SpecialTokens,
}

impl HfTokenizer {
    /// Loads the tokenizer from `tokenizer_json_path` and resolves its special-token ids from the
    /// sibling `tokenizer_config.json` (derived automatically: same directory, file name
    /// `tokenizer_config.json`) plus `tokenizer.json`'s own `added_tokens` / post-processor data —
    /// never hard-coded, since the ids differ by checkpoint.
    ///
    /// This is a convenience wrapper for the fused layout, which always ships a sibling
    /// `tokenizer_config.json`. The split layout (laya-ts's exporter output) does not, and needs
    /// [`Self::from_file_with_config`] with an explicit, possibly-absent config path instead —
    /// [`crate::model_artifacts::ModelArtifacts`] already resolves the right one to pass in.
    ///
    /// Unlike [`Self::from_file_with_config`], a missing sibling `tokenizer_config.json` here
    /// surfaces as [`LayaError::Io`] (a plain "file not found") rather than being wrapped in a
    /// custom message. Callers that want the added_tokens-only fallback even for a sibling that
    /// simply happens to be missing should call `from_file_with_config` directly instead.
    pub fn from_file(tokenizer_json_path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer_json_path = tokenizer_json_path.as_ref();
        let dir = tokenizer_json_path
            .parent()
            .unwrap_or_else(|| Path::new("."));
        let config_path = dir.join("tokenizer_config.json");

        let inner = InnerTokenizer::from_file(tokenizer_json_path).map_err(|e| {
            LayaError::Tokenizer(format!(
                "failed to load tokenizer '{}': {e}",
                tokenizer_json_path.display()
            ))
        })?;
        let special = read_special_tokens(tokenizer_json_path, Some(&config_path))?;
        Ok(Self { inner, special })
    }

    /// Loads the tokenizer from `tokenizer_json_path`, resolving special-token ids from
    /// `tokenizer_config_path` when given, or — when `None`, as for the split layout, which never
    /// ships a `tokenizer_config.json` — falling back to `tokenizer.json` alone: CLS/SEP from its
    /// `post_processor` template, the other roles from its `added_tokens` by matching well-known
    /// content aliases (`[PAD]`/`<pad>`, `[MASK]`/`<mask>`, `[UNK]`/`<unk>`); see
    /// [`resolve_special_tokens_from_added_tokens`].
    pub fn from_file_with_config(
        tokenizer_json_path: impl AsRef<Path>,
        tokenizer_config_path: Option<&Path>,
    ) -> Result<Self> {
        let tokenizer_json_path = tokenizer_json_path.as_ref();
        let inner = InnerTokenizer::from_file(tokenizer_json_path).map_err(|e| {
            LayaError::Tokenizer(format!(
                "failed to load tokenizer '{}': {e}",
                tokenizer_json_path.display()
            ))
        })?;
        let special = read_special_tokens(tokenizer_json_path, tokenizer_config_path)?;
        Ok(Self { inner, special })
    }
}

impl LayaTokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // add_special_tokens = false: the post-processor is never invoked, so no [CLS]/[SEP]/
        // <bos>/<eos> gets injected. SequenceBuilder inserts every special token itself.
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| LayaError::Tokenizer(format!("failed to encode text: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn special(&self) -> &SpecialTokens {
        &self.special
    }
}

// ── special-token resolution ─────────────────────────────────────────────────

/// Reads the special-token ids for the checkpoint whose tokenizer sits at `tokenizer_json_path`.
///
/// When `tokenizer_config_path` is `Some`, this ports `HfTokenizer.ReadSpecialTokenIds` exactly:
/// the config supplies the token *strings* (role -> string), and `tokenizer.json`'s
/// `added_tokens` (falling back to `post_processor.special_tokens`) supplies the string -> id
/// mapping.
///
/// When it is `None` — the split layout never ships a `tokenizer_config.json` — each role's
/// string is instead resolved by matching well-known content aliases directly against
/// `tokenizer.json`'s own `added_tokens` (see [`resolve_special_tokens_from_added_tokens`]),
/// mirroring what a checkpoint's own vocabulary calls its special tokens (mmBERT: `<bos>`/`<eos>`/
/// `<pad>`/`<mask>`/`<unk>`; ModernBERT: `[CLS]`/`[SEP]`/`[PAD]`/`[MASK]`/`[UNK]`) without needing
/// the config file to spell out which string plays which role. CLS and SEP are taken from the
/// `post_processor`'s single-sequence template first, when it has one.
fn read_special_tokens(
    tokenizer_json_path: &Path,
    tokenizer_config_path: Option<&Path>,
) -> Result<SpecialTokens> {
    // Build a content -> id map. Primary: added_tokens enumerates every extended-vocabulary
    // token, including the special ones for both mmBERT (ids 0-4) and ModernBERT (ids
    // 50280-50284). Fallback: post_processor.special_tokens, for a tokenizer that omits a special
    // token from added_tokens.
    let tok_text = std::fs::read_to_string(tokenizer_json_path)?;
    let tok_json: Value = serde_json::from_str(&tok_text)?;

    let mut id_by_content: HashMap<String, u32> = HashMap::new();
    if let Some(added) = tok_json.get("added_tokens").and_then(Value::as_array) {
        for entry in added {
            if let (Some(content), Some(id)) = (
                entry.get("content").and_then(Value::as_str),
                entry.get("id").and_then(Value::as_u64),
            ) {
                id_by_content.insert(content.to_string(), id as u32);
            }
        }
    }
    if let Some(special_tokens) = tok_json
        .get("post_processor")
        .and_then(Value::as_object)
        .and_then(|pp| pp.get("special_tokens"))
        .and_then(Value::as_object)
    {
        for (content, entry) in special_tokens {
            if id_by_content.contains_key(content) {
                continue;
            }
            if let Some(first_id) = entry
                .get("ids")
                .and_then(Value::as_array)
                .and_then(|ids| ids.first())
                .and_then(Value::as_u64)
            {
                id_by_content.insert(content.clone(), first_id as u32);
            }
        }
    }

    match tokenizer_config_path {
        Some(config_path) => {
            // Which string plays each role? A missing file propagates as a plain LayaError::Io.
            let config_text = std::fs::read_to_string(config_path)?;
            let config: Value = serde_json::from_str(&config_text)?;
            let cls_str = require_token_string(&config, "cls_token", config_path)?;
            let sep_str = require_token_string(&config, "sep_token", config_path)?;
            let pad_str = require_token_string(&config, "pad_token", config_path)?;
            let unk_str = require_token_string(&config, "unk_token", config_path)?;
            let mask_str = require_token_string(&config, "mask_token", config_path)?;

            // Resolve each role. Any string missing here means the two files are from different
            // checkpoints, which would silently produce wrong positions if left unchecked.
            Ok(SpecialTokens {
                pad_id: resolve(&pad_str, "pad_token", &id_by_content, config_path)?,
                cls_id: resolve(&cls_str, "cls_token", &id_by_content, config_path)?,
                sep_id: resolve(&sep_str, "sep_token", &id_by_content, config_path)?,
                mask_id: resolve(&mask_str, "mask_token", &id_by_content, config_path)?,
                unk_id: resolve(&unk_str, "unk_token", &id_by_content, config_path)?,
                mask_token: mask_str,
            })
        }
        None => resolve_special_tokens_from_added_tokens(
            &id_by_content,
            template_cls_sep(&tok_json),
            tokenizer_json_path,
        ),
    }
}

/// Reads the CLS and SEP strings from the `post_processor`'s single-sequence template
/// (`[CLS] $A [SEP]`, or `<bos> $A <eos>` for mmBERT): the first and last `SpecialToken` entries.
/// Returns `None` when the tokenizer has no such template.
fn template_cls_sep(tok_json: &Value) -> Option<(String, String)> {
    let specials: Vec<&str> = tok_json
        .get("post_processor")?
        .get("single")?
        .as_array()?
        .iter()
        .filter_map(|piece| piece.get("SpecialToken")?.get("id")?.as_str())
        .collect();
    match specials.as_slice() {
        [first, .., last] => Some((first.to_string(), last.to_string())),
        _ => None,
    }
}

// Alias lists: the content strings a checkpoint's vocabulary is known to use for each special
// role. mmBERT uses `<bos>`/`<eos>`/`<pad>`/`<mask>`/`<unk>`; ModernBERT/BERT-style tokenizers use
// `[CLS]`/`[SEP]`/`[PAD]`/`[MASK]`/`[UNK]`; RoBERTa-style ones use `<s>`/`</s>`. Tried in the order
// listed; the first alias present in `added_tokens` wins. The order matters: mmBERT's vocabulary
// also holds `<s>`/`</s>`, but they are not its CLS/SEP, so `<bos>`/`<eos>` come first. CLS/SEP
// only fall back to these lists when the tokenizer has no post_processor template.
const CLS_ALIASES: &[&str] = &["[CLS]", "<bos>", "<s>"];
const SEP_ALIASES: &[&str] = &["[SEP]", "<eos>", "</s>"];
const PAD_ALIASES: &[&str] = &["[PAD]", "<pad>"];
const MASK_ALIASES: &[&str] = &["[MASK]", "<mask>"];
const UNK_ALIASES: &[&str] = &["[UNK]", "<unk>"];

/// Resolves every special-token role directly from `tokenizer.json`'s own `added_tokens` (already
/// flattened into `id_by_content` by the caller), for a checkpoint with no `tokenizer_config.json`
/// to name which string plays which role. Each role tries its aliases in order and uses the first
/// one present; a role with none of its aliases present is reported by name against
/// `tokenizer_json_path`, so the message points at the actual file that turned out to be missing
/// the token, and — unlike [`resolve`], which is about a stale mismatch between two files — this
/// implies the vocabulary itself is missing an expected special token altogether.
fn resolve_special_tokens_from_added_tokens(
    id_by_content: &HashMap<String, u32>,
    template_cls_sep: Option<(String, String)>,
    tokenizer_json_path: &Path,
) -> Result<SpecialTokens> {
    let (cls_id, sep_id) = match template_cls_sep {
        Some((cls, sep)) => (
            resolve(&cls, "cls_token", id_by_content, tokenizer_json_path)?,
            resolve(&sep, "sep_token", id_by_content, tokenizer_json_path)?,
        ),
        None => (
            resolve_alias(id_by_content, CLS_ALIASES, "cls_token", tokenizer_json_path)?.1,
            resolve_alias(id_by_content, SEP_ALIASES, "sep_token", tokenizer_json_path)?.1,
        ),
    };
    let (mask_token, mask_id) = resolve_alias(
        id_by_content,
        MASK_ALIASES,
        "mask_token",
        tokenizer_json_path,
    )?;
    Ok(SpecialTokens {
        pad_id: resolve_alias(id_by_content, PAD_ALIASES, "pad_token", tokenizer_json_path)?.1,
        cls_id,
        sep_id,
        mask_id,
        unk_id: resolve_alias(id_by_content, UNK_ALIASES, "unk_token", tokenizer_json_path)?.1,
        mask_token,
    })
}

/// Tries each of `aliases` in order against `id_by_content`, returning the first alias string
/// (owned) and id that is present.
fn resolve_alias(
    id_by_content: &HashMap<String, u32>,
    aliases: &[&str],
    role: &str,
    tokenizer_json_path: &Path,
) -> Result<(String, u32)> {
    for alias in aliases {
        if let Some(&id) = id_by_content.get(*alias) {
            return Ok((alias.to_string(), id));
        }
    }
    Err(LayaError::Tokenizer(format!(
        "'{}' has no tokenizer_config.json alongside it, and none of the usual {role} spellings \
         ({}) were found in its added_tokens. Cannot determine the {role} id.",
        tokenizer_json_path.display(),
        aliases.join(", ")
    )))
}

/// Extracts the token string for `key` from a `tokenizer_config.json` root. HuggingFace stores a
/// special token either as a plain string or as a `{"content": "...", ...}` dict; both are
/// handled, matching `HfTokenizer.RequireTokenString`.
fn require_token_string(root: &Value, key: &str, config_path: &Path) -> Result<String> {
    let val = root.get(key).ok_or_else(|| {
        LayaError::Tokenizer(format!(
            "'{}' is missing the '{key}' key — cannot determine the special-token ids.",
            config_path.display()
        ))
    })?;

    match val {
        Value::Object(_) => val
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                LayaError::Tokenizer(format!(
                    "'{key}' in '{}' is a dict with no (non-null) 'content' field.",
                    config_path.display()
                ))
            }),
        Value::String(s) => Ok(s.clone()),
        _ => Err(LayaError::Tokenizer(format!(
            "'{key}' is null (or an unexpected type) in '{}'.",
            config_path.display()
        ))),
    }
}

/// Looks up `token` in `id_by_content`, with a clear error naming the missing token when it is
/// absent — typically meaning the two tokenizer files are from different checkpoints. Ports
/// `HfTokenizer.Resolve`.
fn resolve(
    token: &str,
    role: &str,
    id_by_content: &HashMap<String, u32>,
    config_path: &Path,
) -> Result<u32> {
    id_by_content.get(token).copied().ok_or_else(|| {
        LayaError::Tokenizer(format!(
            "Special token '{token}' (role: {role}, declared in '{}') was not found in \
             tokenizer.json's added_tokens or post_processor.special_tokens. Check that \
             tokenizer.json and tokenizer_config.json belong to the same checkpoint.",
            config_path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny mmBERT-style `tokenizer.json` fixture with the real checkpoint's ids: just enough
    /// `added_tokens` to resolve every special-token role, without a full vocabulary/model section
    /// (unneeded by `read_special_tokens`). Like the real vocabulary it also holds `<s>`/`</s>`,
    /// which are not its CLS/SEP.
    fn mmbert_style_tokenizer_json() -> &'static str {
        r#"{
            "added_tokens": [
                {"id": 0, "content": "<pad>"},
                {"id": 1, "content": "<eos>"},
                {"id": 2, "content": "<bos>"},
                {"id": 3, "content": "<unk>"},
                {"id": 4, "content": "<mask>"},
                {"id": 204, "content": "<s>"},
                {"id": 213, "content": "</s>"}
            ]
        }"#
    }

    fn modernbert_style_tokenizer_json() -> &'static str {
        r#"{
            "added_tokens": [
                {"id": 50281, "content": "[CLS]"},
                {"id": 50282, "content": "[SEP]"},
                {"id": 50283, "content": "[PAD]"},
                {"id": 50284, "content": "[MASK]"},
                {"id": 50280, "content": "[UNK]"}
            ]
        }"#
    }

    #[test]
    fn added_tokens_fallback_resolves_mmbert_style_aliases() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tok_path = dir.path().join("tokenizer.json");
        std::fs::write(&tok_path, mmbert_style_tokenizer_json()).unwrap();

        let special = read_special_tokens(&tok_path, None).expect("should resolve via aliases");
        assert_eq!(special.pad_id, 0);
        assert_eq!(special.sep_id, 1);
        assert_eq!(special.cls_id, 2);
        assert_eq!(special.unk_id, 3);
        assert_eq!(special.mask_id, 4);
        assert_eq!(special.mask_token, "<mask>");
    }

    #[test]
    fn added_tokens_fallback_takes_cls_and_sep_from_the_post_processor_template() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tok_path = dir.path().join("tokenizer.json");
        // The template names "<s> $A </s>", so it must win over the "<bos>"/"<eos>" aliases.
        std::fs::write(
            &tok_path,
            r#"{
                "added_tokens": [
                    {"id": 0, "content": "<pad>"},
                    {"id": 1, "content": "<eos>"},
                    {"id": 2, "content": "<bos>"},
                    {"id": 3, "content": "<unk>"},
                    {"id": 4, "content": "<mask>"},
                    {"id": 204, "content": "<s>"},
                    {"id": 213, "content": "</s>"}
                ],
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [
                        {"SpecialToken": {"id": "<s>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": "</s>", "type_id": 0}}
                    ]
                }
            }"#,
        )
        .unwrap();

        let special = read_special_tokens(&tok_path, None).expect("should resolve via template");
        assert_eq!(special.cls_id, 204);
        assert_eq!(special.sep_id, 213);
        assert_eq!(special.pad_id, 0);
    }

    #[test]
    fn added_tokens_fallback_resolves_modernbert_style_aliases() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tok_path = dir.path().join("tokenizer.json");
        std::fs::write(&tok_path, modernbert_style_tokenizer_json()).unwrap();

        let special = read_special_tokens(&tok_path, None).expect("should resolve via aliases");
        assert_eq!(special.cls_id, 50281);
        assert_eq!(special.sep_id, 50282);
        assert_eq!(special.pad_id, 50283);
        assert_eq!(special.mask_id, 50284);
        assert_eq!(special.unk_id, 50280);
        assert_eq!(special.mask_token, "[MASK]");
    }

    #[test]
    fn added_tokens_fallback_errors_clearly_when_a_role_has_no_alias_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tok_path = dir.path().join("tokenizer.json");
        // Missing a mask token entirely.
        std::fs::write(
            &tok_path,
            r#"{
                "added_tokens": [
                    {"id": 0, "content": "<pad>"},
                    {"id": 1, "content": "<eos>"},
                    {"id": 2, "content": "<bos>"},
                    {"id": 3, "content": "<unk>"}
                ]
            }"#,
        )
        .unwrap();

        let err = read_special_tokens(&tok_path, None).expect_err("must fail: no mask alias");
        let message = err.to_string();
        assert!(
            message.contains("mask_token") && message.contains("tokenizer_config.json"),
            "message was: {message}"
        );
    }

    #[test]
    fn config_based_lookup_still_takes_precedence_when_a_config_path_is_given() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tok_path = dir.path().join("tokenizer.json");
        std::fs::write(&tok_path, mmbert_style_tokenizer_json()).unwrap();

        let config_path = dir.path().join("tokenizer_config.json");
        std::fs::write(
            &config_path,
            r#"{
                "cls_token": "<s>",
                "sep_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "mask_token": "<mask>"
            }"#,
        )
        .unwrap();

        let special =
            read_special_tokens(&tok_path, Some(&config_path)).expect("should resolve via config");
        // The config names "<s>"/"</s>", not the alias-preferred "<bos>"/"<eos>", so it must win.
        assert_eq!(special.cls_id, 204);
        assert_eq!(special.sep_id, 213);
        assert_eq!(special.mask_token, "<mask>");
    }

    #[test]
    fn config_based_lookup_errors_when_the_config_file_is_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tok_path = dir.path().join("tokenizer.json");
        std::fs::write(&tok_path, mmbert_style_tokenizer_json()).unwrap();
        let missing_config = dir.path().join("tokenizer_config.json");

        let err = read_special_tokens(&tok_path, Some(&missing_config))
            .expect_err("must fail: config file does not exist");
        assert!(matches!(err, LayaError::Io(_)));
    }
}
