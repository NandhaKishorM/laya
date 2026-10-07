// Benchmark the Laya .NET SDK across one checkpoint.
//
// Measures:
//   cold latency : engine creation + first Predict (ms)
//   warm latency : median and p95 over N_WARM timed calls (ms), after N_WARMUP warmups
//   peak working set (MB), via Process.WorkingSet64
//   answer snapshot for each bench case
//
// Outputs one JSON object to stdout.
//
// Usage:
//   dotnet run --project samples/Laya.Benchmark -c Release -- --checkpoint multilingual
//   dotnet run --project samples/Laya.Benchmark -c Release -- --checkpoint english
//   dotnet run --project samples/Laya.Benchmark -c Release -- --checkpoint typed-decisions

using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using Laya;

const int N_WARMUP = 5;
const int N_WARM   = 30;

// ── parse args ────────────────────────────────────────────────────────────────
LayaCheckpoint checkpoint = LayaCheckpoint.Multilingual;
string? modelDirectory    = null;
var onnxRoot              = Environment.GetEnvironmentVariable("LAYA_ONNX_ROOT");

for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--checkpoint":
            if (i + 1 >= args.Length) Die("--checkpoint needs a value.");
            checkpoint = args[++i] switch
            {
                "multilingual"    => LayaCheckpoint.Multilingual,
                "english"         => LayaCheckpoint.English,
                "typed-decisions" => LayaCheckpoint.TypedDecisions,
                var s             => DieWith<LayaCheckpoint>($"Unknown checkpoint: {s}"),
            };
            break;
        case "--model-dir":
            if (i + 1 >= args.Length) Die("--model-dir needs a value.");
            modelDirectory = args[++i];
            break;
        default:
            Die($"Unrecognised argument: {args[i]}");
            break;
    }
}

var checkpointName = checkpoint switch
{
    LayaCheckpoint.Multilingual   => "multilingual",
    LayaCheckpoint.English        => "english",
    LayaCheckpoint.TypedDecisions => "typed-decisions",
    _                             => throw new InvalidOperationException(),
};

// If LAYA_ONNX_ROOT is set and no explicit --model-dir was given, derive the dir.
if (modelDirectory is null && onnxRoot is not null)
    modelDirectory = Path.Combine(onnxRoot, checkpointName);

// ── bench cases ───────────────────────────────────────────────────────────────
// Mirrors laya-dotnet/tools/bench_python.py: the same two states and question set.
var englishState = new Dictionary<string, object?>
{
    ["from"]    = "user@acme.com",
    ["subject"] = "Duplicate charge on invoice #4411",
    ["body"]    = "Hi, we were billed twice for March. Please refund the duplicate today "
                + "or we will cancel our plan.",
};
var hindiState = new Dictionary<string, object?>
{
    ["body"] = "मार्च का बिल "
             + "दो बार लिया गया, "
             + "कृपया रिफंड करें।",
};
var questions = new QuestionSet
{
    ["department"]       = Question.Choice(
        "Which department should handle this request?",
        ("billing",   "invoices, payments, refunds"),
        ("technical", "bugs, outages, system errors"),
        ("sales",     "pricing, new contracts"),
        ("other",     "everything else")),
    ["urgency"]          = Question.Score(
        "How urgent is this request?",
        "not urgent", "soon", "critical deadline or blocking issue"),
    ["churn_risk"]       = Question.Noul("Does the user threaten to cancel or leave?"),
    ["refund_requested"] = Question.Noul("Does the user explicitly request a refund?"),
};
var benchCases = new (string Name, object? State)[]
{
    ("sample_app_english", englishState),
    ("sample_app_hindi",   hindiState),
};

// ── cold measurement ──────────────────────────────────────────────────────────
var coldClock = Stopwatch.StartNew();
LayaEngine engine;
try
{
    engine = LayaEngine.Create(new LayaOptions
    {
        Checkpoint      = checkpoint,
        ModelDirectory  = modelDirectory,
    });
}
catch (Exception ex) when (ex is DirectoryNotFoundException or FileNotFoundException)
{
    Console.Error.WriteLine(ex.Message);
    return 1;
}

// First call (still part of cold, as ORT JITs on the first run).
var (firstName, firstState) = benchCases[0];
engine.Predict(firstState, questions);
coldClock.Stop();
var coldMs = coldClock.Elapsed.TotalMilliseconds;

Console.Error.WriteLine($"[bench_dotnet] cold={coldMs:F0} ms, running warm-up...");

// ── warmup ────────────────────────────────────────────────────────────────────
var callIdx = 0;
(object? State, QuestionSet Qs) NextCall()
{
    var c = benchCases[callIdx++ % benchCases.Length];
    return (c.State, questions);
}

for (var i = 0; i < N_WARMUP; i++)
{
    var (s, q) = NextCall();
    engine.Predict(s, q);
}

// ── warm measurement ──────────────────────────────────────────────────────────
var timesMs = new double[N_WARM];
for (var i = 0; i < N_WARM; i++)
{
    var (s, q) = NextCall();
    var t0 = Stopwatch.GetTimestamp();
    engine.Predict(s, q);
    timesMs[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
}
Array.Sort(timesMs);
var medianMs = timesMs[timesMs.Length / 2];
var p95Ms    = timesMs[(int)(timesMs.Length * 0.95)];

// ── peak working set ──────────────────────────────────────────────────────────
var peakMb = Process.GetCurrentProcess().PeakWorkingSet64 / (1024.0 * 1024.0);

Console.Error.WriteLine(
    $"[bench_dotnet] warm median={medianMs:F1} ms  p95={p95Ms:F1} ms  peak={peakMb:F0} MB");

// ── answer snapshot ───────────────────────────────────────────────────────────
var answers = new Dictionary<string, object>();
foreach (var (name, state) in benchCases)
{
    var result = engine.Predict(state, questions);
    var caseAnswers = new Dictionary<string, object>();
    foreach (var id in result.Ids)
    {
        var a = result[id];
        switch (a)
        {
            case ChoiceAnswer choice:
                caseAnswers[id] = new
                {
                    type           = "choice",
                    choice         = choice.Choice,
                    probabilities  = choice.Probabilities.ToDictionary(p => p.Key, p => p.Value),
                    confidence     = choice.Confidence,
                    act_probability = choice.Action.ActProbability,
                };
                break;
            case ScoreAnswer score:
                caseAnswers[id] = new
                {
                    type           = "score",
                    score          = score.Score,
                    probabilities  = score.Probabilities.Select((p, i) => (i.ToString(), p))
                                         .ToDictionary(x => x.Item1, x => x.p),
                    confidence     = score.Confidence,
                    act_probability = score.Action.ActProbability,
                };
                break;
            case NoulAnswer noul:
                caseAnswers[id] = new
                {
                    type           = "noul",
                    noul           = noul.Probability,
                    confidence     = noul.Confidence,
                    act_probability = noul.Action.ActProbability,
                };
                break;
        }
    }
    answers[name] = caseAnswers;
}

engine.Dispose();

// ── emit JSON ─────────────────────────────────────────────────────────────────
var output = new
{
    checkpoint      = checkpointName,
    runtime         = "DotNet + ONNX Runtime",
    threads         = "ORT default",
    cold_ms         = Math.Round(coldMs, 1),
    warm_times_ms   = timesMs.Select(t => Math.Round(t, 2)).ToArray(),
    warm_median_ms  = Math.Round(medianMs, 1),
    warm_p95_ms     = Math.Round(p95Ms, 1),
    n_warmup        = N_WARMUP,
    n_warm          = N_WARM,
    peak_wset_mb    = Math.Round(peakMb, 1),
    answers,
};

var json = JsonSerializer.Serialize(output, new JsonSerializerOptions
{
    WriteIndented  = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.Never,
});
Console.WriteLine(json);
return 0;

// ── helpers ───────────────────────────────────────────────────────────────────
static void Die(string msg) { Console.Error.WriteLine(msg); Environment.Exit(2); }
static T DieWith<T>(string msg) { Die(msg); return default!; }
