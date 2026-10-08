//! The README quickstart, run natively on ONNX Runtime with no Python in the process.
//!
//! The artifact directory is resolved from `LayaOptions::model_directory`, then `LAYA_ONNX_DIR`,
//! then the local cache. Point `LAYA_ONNX_DIR` at a directory holding `model.onnx`,
//! `model.onnx.data`, `rl_agent_config.json` and `tokenizer/tokenizer.json`, or pass `--download`
//! to fetch them once.
//!
//! Mirrors the Python SDK's quickstart in the repo root README (arguments, output, exit codes)
//! one to one, running natively on ONNX Runtime instead.

use laya::{
    Answer, ArtifactDownloadProgress, LayaCheckpoint, LayaEngine, LayaError, LayaOptions, Question,
    QuestionSet, Result as LayaResultT,
};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
Usage: cargo run --release -p laya-sample -- [options]

  --model-dir <dir>           Directory holding model.onnx, model.onnx.data, rl_agent_config.json
                              and tokenizer/tokenizer.json. Overrides LAYA_ONNX_DIR.
  --checkpoint <name>         Checkpoint to load: multilingual (default), english, typed-decisions.
  --download                  Fetch the artifact from Hugging Face into the local cache.
                              Requires the ONNX export to be published there.
  -h, --help                  Print this message.

With no options the artifact is resolved from LAYA_ONNX_DIR, then the local cache.";

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut allow_download = false;
    let mut model_directory: Option<String> = None;
    let mut checkpoint = LayaCheckpoint::Multilingual;

    // Parsed by hand rather than ignoring what is not recognised: `--download <path>` reads as
    // though the path were the cache location, and silently dropping it sends the run somewhere
    // the user did not ask for -- which is a confusing way to then fail on the network.
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--download" => {
                allow_download = true;
            }

            "--model-dir" => {
                if i + 1 >= args.len() {
                    eprintln!("--model-dir needs a directory.");
                    eprintln!("{USAGE}");
                    return 2;
                }
                i += 1;
                model_directory = Some(args[i].clone());
            }

            "--checkpoint" => {
                if i + 1 >= args.len() {
                    eprintln!(
                        "--checkpoint needs a name (multilingual, english, typed-decisions)."
                    );
                    eprintln!("{USAGE}");
                    return 2;
                }
                i += 1;
                let name = args[i].as_str();
                checkpoint = match name {
                    "multilingual" => LayaCheckpoint::Multilingual,
                    "english" => LayaCheckpoint::English,
                    "typed-decisions" => LayaCheckpoint::TypedDecisions,
                    _ => {
                        eprintln!(
                            "Unknown checkpoint '{name}'. Valid values: multilingual, english, typed-decisions."
                        );
                        eprintln!("{USAGE}");
                        return 2;
                    }
                };
            }

            "-h" | "--help" => {
                println!("{USAGE}");
                return 0;
            }

            other => {
                eprintln!("Unrecognised argument: {other}");
                eprintln!("{USAGE}");
                return 2;
            }
        }
        i += 1;
    }

    let mut options = LayaOptions::new();
    options.model_directory = model_directory.map(PathBuf::from);
    options.checkpoint = checkpoint;
    options.allow_download = allow_download;
    if allow_download {
        options.download_progress = Some(Arc::new(|p: ArtifactDownloadProgress| {
            eprintln!(
                "  {}: {}/{}",
                p.file_name,
                p.bytes_received,
                p.total_bytes
                    .map(|b| b.to_string())
                    .unwrap_or_else(|| "?".to_string())
            );
        }));
    }

    // A directory that exists but is missing one file (the 1.29 GB model.onnx.data especially,
    // which is easy to leave behind when copying an export) is a misconfiguration like any other,
    // and the resolver's message already explains it better than a stack trace does. These three
    // variants together cover every "artifacts missing" outcome the resolver can report.
    let engine = match LayaEngine::create(options) {
        Ok(engine) => engine,
        Err(
            ref e @ (LayaError::ArtifactsNotFound { .. }
            | LayaError::DirectoryNotFound { .. }
            | LayaError::MissingFile { .. }),
        ) => {
            // The resolver's message already lists every location it looked in, so print it and
            // stop rather than burying it in a stack trace.
            eprintln!("{e}");
            if !allow_download {
                // Only suggested when a download was not already tried: the same exception
                // carries the download's own failure, and "try --download" under a --download
                // failure reads as nonsense.
                eprintln!();
                eprintln!("Point --model-dir or LAYA_ONNX_DIR at a local ONNX export.");
            }
            return 1;
        }
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    // 1. State in any language or schema.
    let state = json!({
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today \
                 or we will cancel our plan.",
    });

    // 2. Typed questions. Choice labels are positional, so the option order here is the order the
    //    model scores them in -- QuestionSet and Question::choice both preserve it.
    let questions = match build_questions() {
        Ok(q) => q,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    // 3. One forward pass answers every question.
    report(&engine, state, &questions, "English");

    // 4. The multilingual checkpoint covers 100+ languages, so the same questions work unchanged.
    let hindi = json!({
        "body": "मार्च का बिल दो बार लिया गया, कृपया रिफंड करें।",
    });
    report(&engine, hindi, &questions, "Hindi");

    0
}

fn build_questions() -> LayaResultT<QuestionSet> {
    QuestionSet::new()
        .with(
            "department",
            Question::choice(
                "Which department should handle this request?",
                [
                    ("billing", "invoices, payments, refunds"),
                    ("technical", "bugs, outages, system errors"),
                    ("sales", "pricing, new contracts"),
                    ("other", "everything else"),
                ],
            )?,
        )?
        .with(
            "urgency",
            Question::score(
                "How urgent is this request?",
                ["not urgent", "soon", "critical deadline or blocking issue"],
            )?,
        )?
        .with(
            "churn_risk",
            Question::noul(
                "Does the user threaten to cancel or leave?",
                Value::Null,
                Value::Null,
            ),
        )?
        .with(
            "refund_requested",
            Question::noul(
                "Does the user explicitly request a refund?",
                Value::Null,
                Value::Null,
            ),
        )
}

fn report(engine: &LayaEngine, state: Value, questions: &QuestionSet, label: &str) {
    // First call pays for the ORT session warm-up, so both timings are printed.
    let clock = Instant::now();
    let result = match engine.predict(state, questions) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    let elapsed_ms = clock.elapsed().as_secs_f64() * 1000.0;

    println!();
    println!("── {label} ─────────────────────────────────");
    println!("model      : {}", result.model());
    println!(
        "latency    : {elapsed_ms:.0} ms  ({} input tokens, {} output tokens)",
        result.usage().input_tokens,
        result.usage().output_tokens
    );

    // Indexing the result by question id gives the base Answer; as_* narrows it to the typed
    // shape.
    for id in result.ids() {
        let answer = &result[id.as_str()];
        let value = match answer {
            Answer::Choice(choice) => choice.choice.clone(),
            Answer::Score(score) => format!("{:.2}", score.score),
            Answer::Noul(noul) => if noul.value() { "yes" } else { "no" }.to_string(),
        };
        println!(
            "{id:<17}: {value:<8} (confidence {:.2}, act {:.2})",
            answer.confidence(),
            answer.action().act_probability
        );
    }

    // The full distribution is there when a single label is not enough to act on.
    let department = result["department"]
        .as_choice()
        .expect("department is a choice question");
    let distribution: Vec<String> = department
        .probabilities
        .iter()
        .map(|(label, prob)| format!("{label} {:.1} %", prob * 100.0))
        .collect();
    println!("  department distribution: {}", distribution.join(", "));
}
