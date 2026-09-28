//! Access to per-checkpoint parity vectors recorded by `tools/dump_golden.py`.
//!
//! Golden files are read from the source tree rather than copied anywhere: they are the
//! reference, and a stale copy beside the test binary would let a real divergence pass.
//!
//! There is no integer-vs-float conversion step needed here: `serde_json::Value` already
//! preserves the distinction between a JSON integer and a JSON float (a `3` in the source JSON
//! parses to an integer `Number`, `3.0` to a float `Number`), so a parsed `Value` can be handed
//! straight to [`laya::questions::Question`] constructors.

use laya::error::Result as LayaResult;
use laya::questions::{Question, QuestionSet};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The base `golden/` directory (parent of the per-checkpoint subdirectories), overridable with
/// `LAYA_GOLDEN_DIR`. Default: `$CARGO_MANIFEST_DIR/tests/golden`.
pub fn golden_base_dir() -> PathBuf {
    match std::env::var("LAYA_GOLDEN_DIR") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden"),
    }
}

/// The golden directory for one checkpoint subfolder (`"multilingual"`, `"english"`,
/// `"typed-decisions"`).
pub fn checkpoint_dir(subdir: &str) -> PathBuf {
    golden_base_dir().join(subdir)
}

/// One entry of `index.json`.
#[derive(Debug, Clone)]
pub struct GoldenCaseInfo {
    /// The case's short name, e.g. `"quickstart"`.
    pub name: String,
    /// The file it is recorded in, e.g. `"case_quickstart.json"`.
    pub file: String,
    /// Whether Python raised an exception for this case rather than returning a result.
    pub expects_error: bool,
    /// The entry's `"kind"` field, or `None` when absent (a standard predict-parity case). The
    /// routing port added a `"shortlist"` kind (`shortlist_many_options`); a case with a `kind`
    /// other than the standard one is not a plain `Agent.predict` recording and must be excluded
    /// from [`CheckpointGoldenData::success_cases`] / [`CheckpointGoldenData::error_cases`], or
    /// the ordinary parity tests try to feed it through `Question` construction and fail on
    /// shape, not on a real regression.
    pub kind: Option<String>,
}

impl GoldenCaseInfo {
    /// Whether this is a standard predict-parity case (no `"kind"`, or `"kind": "standard"`).
    fn is_standard(&self) -> bool {
        matches!(self.kind.as_deref(), None | Some("standard"))
    }
}

impl std::fmt::Display for GoldenCaseInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Access to one checkpoint's recorded golden vectors.
pub struct CheckpointGoldenData {
    dir: PathBuf,
}

impl CheckpointGoldenData {
    /// Golden data for the checkpoint subfolder named `subdir`.
    pub fn for_checkpoint(subdir: &str) -> Self {
        Self {
            dir: checkpoint_dir(subdir),
        }
    }

    /// Whether any golden data was found for this checkpoint.
    pub fn available(&self) -> bool {
        self.dir.join("index.json").is_file()
    }

    /// Parse one golden file.
    pub fn load(&self, file_name: &str) -> Value {
        let path = self.dir.join(file_name);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read golden file {}: {e}", path.display()));
        serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("failed to parse golden file {}: {e}", path.display()))
    }

    /// The recording environment: versions, config, and the special token ids.
    pub fn meta(&self) -> Value {
        self.load("meta.json")
    }

    /// Every recorded case, in the order the dump script emitted them.
    pub fn index(&self) -> Vec<GoldenCaseInfo> {
        let doc = self.load("index.json");
        doc.as_array()
            .unwrap_or_else(|| panic!("{}/index.json must be a JSON array", self.dir.display()))
            .iter()
            .map(|e| GoldenCaseInfo {
                name: e["name"]
                    .as_str()
                    .expect("index entry missing 'name'")
                    .to_string(),
                file: e["file"]
                    .as_str()
                    .expect("index entry missing 'file'")
                    .to_string(),
                expects_error: e["expects_error"]
                    .as_bool()
                    .expect("index entry missing 'expects_error'"),
                kind: e["kind"].as_str().map(str::to_string),
            })
            .collect()
    }

    /// Standard cases (see [`GoldenCaseInfo::is_standard`]) that record a successful prediction.
    pub fn success_cases(&self) -> Vec<GoldenCaseInfo> {
        self.index()
            .into_iter()
            .filter(|c| c.is_standard() && !c.expects_error)
            .collect()
    }

    /// Standard cases that record a Python exception.
    pub fn error_cases(&self) -> Vec<GoldenCaseInfo> {
        self.index()
            .into_iter()
            .filter(|c| c.is_standard() && c.expects_error)
            .collect()
    }

    /// Cases recorded with `"kind": "shortlist"` (`shortlist_many_options`): a hashing-embedder
    /// reduced choice question run through the ordinary predict pipeline, not a plain
    /// `Agent.predict` recording, so it is kept out of [`Self::success_cases`].
    pub fn shortlist_cases(&self) -> Vec<GoldenCaseInfo> {
        self.index()
            .into_iter()
            .filter(|c| c.kind.as_deref() == Some("shortlist"))
            .collect()
    }

    /// Parse the file for one case.
    pub fn load_case(&self, info: &GoldenCaseInfo) -> Value {
        self.load(&info.file)
    }
}

/// Rebuild a case's [`QuestionSet`] from its recorded `questions` definition. Ports
/// `GoldenData.BuildQuestions` / `BuildChoice` / `BuildNoul`.
pub fn build_questions(questions: &Value) -> LayaResult<QuestionSet> {
    let obj = questions
        .as_object()
        .expect("'questions' must be a JSON object");
    let mut set = QuestionSet::new();
    for (id, def) in obj {
        let qtype = def
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("question '{id}' has no 'type'"));
        let instructions = def
            .get("instructions")
            .cloned()
            .filter(|v| !v.is_null())
            .unwrap_or_else(|| panic!("question '{id}' has null instructions"));

        let question = match qtype {
            "choice" => build_choice(id, instructions, def.get("criteria")),
            "score" => {
                let levels = def
                    .get("criteria")
                    .and_then(Value::as_array)
                    .unwrap_or_else(|| panic!("question '{id}' needs a criteria array"));
                Question::score(instructions, levels.iter().cloned())
                    .unwrap_or_else(|e| panic!("question '{id}': {e}"))
            }
            "noul" => build_noul(instructions, def.get("criteria")),
            other => panic!("question '{id}' has unknown type '{other}'"),
        };
        set.insert(id.clone(), question)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    Ok(set)
}

fn build_choice(id: &str, instructions: Value, criteria: Option<&Value>) -> Question {
    let criteria = criteria.unwrap_or_else(|| panic!("question '{id}' has no 'criteria'"));
    match criteria {
        // Python normalizes a bare list of labels to {label: None} in Agent._to_internal.
        Value::Array(labels) => {
            let opts: Vec<(String, Value)> = labels
                .iter()
                .map(|v| {
                    let label = v
                        .as_str()
                        .unwrap_or_else(|| panic!("question '{id}' has a non-string label"))
                        .to_string();
                    (label, Value::Null)
                })
                .collect();
            Question::choice(instructions, opts).unwrap_or_else(|e| panic!("question '{id}': {e}"))
        }
        Value::Object(map) => {
            let opts: Vec<(String, Value)> =
                map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            Question::choice(instructions, opts).unwrap_or_else(|e| panic!("question '{id}': {e}"))
        }
        _ => panic!("question '{id}' has unusable criteria"),
    }
}

fn build_noul(instructions: Value, criteria: Option<&Value>) -> Question {
    match criteria {
        Some(Value::Object(map)) => {
            let if_false = map.get("false").cloned().unwrap_or(Value::Null);
            let if_true = map.get("true").cloned().unwrap_or(Value::Null);
            Question::noul(instructions, if_false, if_true)
        }
        _ => Question::noul(instructions, Value::Null, Value::Null),
    }
}

/// A recorded `[rows][cols]` float matrix, flattened row-major.
pub fn matrix(element: &Value, rows: &mut usize, cols: &mut usize) -> Vec<f32> {
    let arr = element.as_array().expect("expected a JSON array");
    *rows = arr.len();
    *cols = if arr.is_empty() {
        0
    } else {
        arr[0].as_array().map_or(0, Vec::len)
    };
    let mut flat = Vec::with_capacity(rows.saturating_mul(*cols));
    for row in arr {
        for value in row.as_array().expect("expected a row array") {
            flat.push(value.as_f64().expect("expected a number") as f32);
        }
    }
    flat
}

/// A recorded integer array.
pub fn ints(element: &Value) -> Vec<i64> {
    element
        .as_array()
        .expect("expected a JSON array")
        .iter()
        .map(|e| e.as_i64().expect("expected an integer"))
        .collect()
}

/// A recorded integer array, as `usize` (for marker positions).
pub fn indices(element: &Value) -> Vec<usize> {
    ints(element).into_iter().map(|v| v as usize).collect()
}
