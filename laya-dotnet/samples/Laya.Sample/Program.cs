using System.Diagnostics;
using Laya;

// The README quickstart, run natively on ONNX Runtime with no Python in the process.
//
// The artifact directory is resolved from LayaOptions.ModelDirectory, then LAYA_ONNX_DIR, then the
// local cache. Point LAYA_ONNX_DIR at a directory holding model.onnx, model.onnx.data,
// rl_agent_config.json and tokenizer/tokenizer.json, or pass --download to fetch them once.

const string Usage = """
    Usage: dotnet run --project samples/Laya.Sample [options]

      --model-dir <dir>           Directory holding model.onnx, model.onnx.data, rl_agent_config.json
                                  and tokenizer/tokenizer.json. Overrides LAYA_ONNX_DIR.
      --checkpoint <name>         Checkpoint to load: multilingual (default), english, typed-decisions.
      --download                  Fetch the artifact from Hugging Face into the local cache.
                                  Requires the ONNX export to be published there.
      -h, --help                  Print this message.

    With no options the artifact is resolved from LAYA_ONNX_DIR, then the local cache.
    """;

var allowDownload = false;
string? modelDirectory = null;
LayaCheckpoint checkpoint = LayaCheckpoint.Multilingual;

// Parsed by hand rather than ignoring what is not recognised: `--download <path>` reads as though the
// path were the cache location, and silently dropping it sends the run somewhere the user did not ask
// for -- which is a confusing way to then fail on the network.
for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--download":
            allowDownload = true;
            break;

        case "--model-dir":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("--model-dir needs a directory.");
                Console.Error.WriteLine(Usage);
                return 2;
            }
            modelDirectory = args[++i];
            break;

        case "--checkpoint":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("--checkpoint needs a name (multilingual, english, typed-decisions).");
                Console.Error.WriteLine(Usage);
                return 2;
            }
            var name = args[++i];
            checkpoint = name switch
            {
                "multilingual"   => LayaCheckpoint.Multilingual,
                "english"        => LayaCheckpoint.English,
                "typed-decisions" => LayaCheckpoint.TypedDecisions,
                _ => (LayaCheckpoint)(-1), // sentinel for unrecognised
            };
            if ((int)checkpoint == -1)
            {
                Console.Error.WriteLine(
                    $"Unknown checkpoint '{name}'. Valid values: multilingual, english, typed-decisions.");
                Console.Error.WriteLine(Usage);
                return 2;
            }
            break;

        case "-h":
        case "--help":
            Console.WriteLine(Usage);
            return 0;

        default:
            Console.Error.WriteLine($"Unrecognised argument: {args[i]}");
            Console.Error.WriteLine(Usage);
            return 2;
    }
}

LayaEngine engine;
try
{
    engine = LayaEngine.Create(new LayaOptions
    {
        ModelDirectory = modelDirectory,
        Checkpoint     = checkpoint,
        AllowDownload  = allowDownload,
        DownloadProgress = allowDownload
            ? new Progress<ArtifactDownloadProgress>(p => Console.Error.WriteLine($"  {p}"))
            : null,
    });
}
// FileNotFoundException is caught alongside it: a directory that exists but is missing one file (the
// 1.29 GB model.onnx.data especially, which is easy to leave behind when copying an export) is a
// misconfiguration like any other, and the resolver's message already explains it better than a
// stack trace does.
catch (Exception error) when (error is DirectoryNotFoundException or FileNotFoundException)
{
    // The resolver's message already lists every location it looked in, so print it and stop
    // rather than burying it in a stack trace.
    Console.Error.WriteLine(error.Message);
    if (!allowDownload)
    {
        // Only suggested when a download was not already tried: the same exception carries the
        // download's own failure, and "try --download" under a --download failure reads as nonsense.
        Console.Error.WriteLine();
        Console.Error.WriteLine("Point --model-dir or LAYA_ONNX_DIR at a local ONNX export.");
    }
    return 1;
}

using (engine)
{
    // 1. State in any language or schema.
    var state = new Dictionary<string, object?>
    {
        ["from"] = "user@acme.com",
        ["subject"] = "Duplicate charge on invoice #4411",
        ["body"] = "Hi, we were billed twice for March. Please refund the duplicate today "
                 + "or we will cancel our plan.",
    };

    // 2. Typed questions. Choice labels are positional, so the option order here is the order the
    //    model scores them in -- QuestionSet and Question.Choice both preserve it.
    var questions = new QuestionSet
    {
        ["department"] = Question.Choice(
            "Which department should handle this request?",
            ("billing", "invoices, payments, refunds"),
            ("technical", "bugs, outages, system errors"),
            ("sales", "pricing, new contracts"),
            ("other", "everything else")),
        ["urgency"] = Question.Score(
            "How urgent is this request?",
            "not urgent", "soon", "critical deadline or blocking issue"),
        ["churn_risk"] = Question.Noul("Does the user threaten to cancel or leave?"),
        ["refund_requested"] = Question.Noul("Does the user explicitly request a refund?"),
    };

    // 3. One forward pass answers every question.
    Report(engine, state, questions, "English");

    // 4. The multilingual checkpoint covers 100+ languages, so the same questions work unchanged.
    var hindi = new Dictionary<string, object?>
    {
        ["body"] = "मार्च का बिल "
                 + "दो बार लिया गया, "
                 + "कृपया रिफंड करें।",
    };
    Report(engine, hindi, questions, "Hindi");
}

return 0;

static void Report(LayaEngine engine, object? state, QuestionSet questions, string label)
{
    // First call pays for the ORT session warm-up, so both timings are printed.
    var clock = Stopwatch.StartNew();
    var result = engine.Predict(state, questions);
    clock.Stop();

    Console.WriteLine();
    Console.WriteLine($"── {label} ─────────────────────────────────");
    Console.WriteLine($"model      : {result.Model}");
    Console.WriteLine($"latency    : {clock.Elapsed.TotalMilliseconds:F0} ms  "
                    + $"({result.Usage.InputTokens} input tokens, "
                    + $"{result.Usage.OutputTokens} output tokens)");

    // Indexing the result by question id gives the base Answer; As* narrows it to the typed shape.
    foreach (var id in result.Ids)
    {
        var answer = result[id];
        var value = answer switch
        {
            ChoiceAnswer choice => choice.Choice,
            ScoreAnswer score => score.Score.ToString("F2"),
            NoulAnswer noul => noul.Value ? "yes" : "no",
            _ => "?",
        };
        Console.WriteLine($"{id,-17}: {value,-8} (confidence {answer.Confidence:F2}, "
                        + $"act {answer.Action.ActProbability:F2})");
    }

    // The full distribution is there when a single label is not enough to act on.
    var department = result["department"].AsChoice();
    Console.WriteLine("  department distribution: " + string.Join(", ",
        department.Probabilities.Select(p => $"{p.Key} {p.Value:P1}")));
}
