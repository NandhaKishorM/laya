using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Model-free tests of calibration files and histogram binning: <c>apply_calibration_payload</c> and
/// <c>apply_binning_map</c> against <c>golden/routing/calibration_probe.json</c> (Python's own functions over the
/// synthetic inputs of <c>tools/calibration_cases.py</c>), the loader's error texts, and the decode of one row over synthetic logits. The decode over the real
/// weights is pinned by the <c>calibration_*</c> cases of <see cref="PredictParityTests"/>.
/// </summary>
public sealed class CalibrationFileTests
{
    // ── apply_calibration_payload ─────────────────────────────────────────────

    private static readonly Lazy<JsonDocument> Probe = new(() => RoutingGoldenData.Load("calibration_probe.json"));

    private static JsonElement Payloads => Probe.Value.RootElement.GetProperty("payloads");

    private static JsonElement Lookups => Probe.Value.RootElement.GetProperty("lookups");

    public static IEnumerable<object[]> PayloadCases() => Enumerable.Range(0, Payloads.GetArrayLength()).Select(i => new object[] { i });

    public static IEnumerable<object[]> LookupCases() => Enumerable.Range(0, Lookups.GetArrayLength()).Select(i => new object[] { i });

    [Fact]
    public void TheProbeCoversAcceptedPayloadsRefusalsAndWarnings()
    {
        Assert.True(Payloads.GetArrayLength() > 140);
        Assert.Contains(PayloadCases(), c => !Payloads[(int)c[0]].TryGetProperty("error_message", out _));
        Assert.Contains(PayloadCases(), c => Payloads[(int)c[0]].TryGetProperty("error_message", out _));
        Assert.Contains(PayloadCases(), c => Payloads[(int)c[0]].GetProperty("warnings").GetArrayLength() > 0);
        Assert.Equal(2, Probe.Value.RootElement.GetProperty("version").GetInt32());
        Assert.Equal(Calibration.TempMin, Probe.Value.RootElement.GetProperty("temp_min").GetDouble());
        Assert.Equal(Calibration.TempMax, Probe.Value.RootElement.GetProperty("temp_max").GetDouble());
    }

    private const string IdentityWarningPrefix = "calibration was fitted for model_id_or_path=";

    [Theory]
    [MemberData(nameof(PayloadCases))]
    public void AppliesAPayloadLikePython(int index)
    {
        var expected = Payloads[index];
        var label = expected.GetProperty("label").GetString();
        var payload = PythonJsonReader.Parse(expected.GetProperty("json").GetString()!);

        CalibrationPayload? got = null;
        InvalidDataException? refused = null;
        List<string> warnings = [];
        try { got = CalibrationFile.Apply(payload, warnings.Add); }
        catch (InvalidDataException e) { refused = e; }

        // The probe also records Python's checkpoint-identity warning (`_warn_if_identity_mismatch`, a version 2 file whose
        // recorded model_id_or_path/subfolder/config differs from the agent's). This SDK has no equivalent of Python's
        // model_id_or_path (the string its caller passed), so it does not emit it; every other warning text is asserted.
        Assert.Equal(
            [.. expected.GetProperty("warnings").EnumerateArray().Select(w => w.GetProperty("message").GetString()!)
                .Where(m => !m.StartsWith(IdentityWarningPrefix, StringComparison.Ordinal))],
            warnings);

        if (expected.TryGetProperty("error_message", out var message))
        {
            Assert.True(refused is not null, $"{label}: expected {expected.GetProperty("error_type").GetString()}");
            Assert.Equal($"{expected.GetProperty("error_type").GetString()}: {message.GetString()}", refused!.Message);
            return;
        }
        Assert.True(refused is null, $"{label}: {refused?.Message}");

        var result = expected.GetProperty("result");
        Assert.Equal([.. result.GetProperty("temperature").EnumerateArray().Select(CallOptions.Number)], got!.Temperature);
        Assert.Equal(
            result.GetProperty("temperature_by_options").EnumerateObject().Select(p => (p.Name, CallOptions.Number(p.Value))),
            got.TemperatureByOptions.Select(kv => (kv.Key, kv.Value)));
        var binning = result.GetProperty("binning_map");
        if (binning.ValueKind == JsonValueKind.Null)
        {
            Assert.Null(got.BinningMap);
            return;
        }
        Assert.NotNull(got.BinningMap);
        Assert.Equal(binning.EnumerateObject().Select(p => p.Name), got.BinningMap!.Keys);
        foreach (var entry in binning.EnumerateObject())
        {
            Assert.Equal(entry.Value.GetProperty("bins").GetInt32(), got.BinningMap[entry.Name].Bins);
            Assert.Equal(
                [.. entry.Value.GetProperty("values").EnumerateArray().Select(CallOptions.Number)],
                got.BinningMap[entry.Name].Values);
        }
    }

    // ── apply_binning_map ─────────────────────────────────────────────────────

    private static IReadOnlyDictionary<string, BinningEntry> Map(string name)
    {
        var map = new Dictionary<string, BinningEntry>(StringComparer.Ordinal);
        foreach (var entry in Probe.Value.RootElement.GetProperty("maps").GetProperty(name).EnumerateObject())
            map[entry.Name] = new BinningEntry(
                entry.Value.GetProperty("bins").GetInt32(),
                [.. entry.Value.GetProperty("values").EnumerateArray().Select(CallOptions.Number)]);
        return map;
    }

    [Fact]
    public void TheLookupProbeCoversEdgesMissesAndRefusals()
    {
        Assert.True(Lookups.GetArrayLength() > 60);
        var labels = Enumerable.Range(0, Lookups.GetArrayLength()).Select(i => Lookups[i].GetProperty("label").GetString()!).ToList();
        Assert.Contains("choice:3-5 at 0.0", labels);
        Assert.Contains("choice:3-5 at 1.0", labels);
        Assert.Contains(labels, l => l.StartsWith("miss at", StringComparison.Ordinal));
        Assert.Contains(Enumerable.Range(0, Lookups.GetArrayLength()), i => Lookups[i].TryGetProperty("error_message", out _));
    }

    [Theory]
    [MemberData(nameof(LookupCases))]
    public void BinsAConfidenceLikePython(int index)
    {
        var expected = Lookups[index];
        var map = Map(expected.GetProperty("map").GetString()!);
        var confidence = CallOptions.Number(expected.GetProperty("confidence"));
        var bucket = expected.GetProperty("bucket").GetString()!;

        if (expected.TryGetProperty("error_message", out var message))
        {
            var error = Assert.Throws<InvalidDataException>(() => CalibrationFile.ApplyBinning(confidence, bucket, map));
            Assert.Equal($"{expected.GetProperty("error_type").GetString()}: {message.GetString()}", error.Message);
            return;
        }
        Assert.Equal(CallOptions.Number(expected.GetProperty("result")), CalibrationFile.ApplyBinning(confidence, bucket, map));
    }

    [Fact]
    public void NoMapIsNoBinning()
    {
        Assert.Equal(0.37, CalibrationFile.ApplyBinning(0.37, "choice:3-5", null));
    }

    // ── the shared fixture ────────────────────────────────────────────────────

    [Fact]
    public void TheFixtureLoadsAndHoldsSeveralBuckets()
    {
        var loaded = CalibrationFile.Load(CalibrationFixture.Path);
        Assert.Equal([1.8, 2.2, 1.3], loaded.Temperature);
        Assert.Equal(2.4, loaded.TemperatureByOptions["choice:3-5"]);
        Assert.Equal(0.6, loaded.TemperatureByOptions["choice:6-10"]);
        Assert.NotNull(loaded.BinningMap);
        Assert.Equal(["choice:3-5", "noul:2", "score:6-10", "choice:11+", "score:2"], loaded.BinningMap!.Keys);
        Assert.Equal(4, loaded.BinningMap["choice:3-5"].Bins);
    }

    // ── the loader and its error texts, without a model ───────────────────────
    // Python loads the file last, after the session and the tokenizer (onnx_agent.py:263), so `LayaEngine.Create` reaches it only
    // over real artifacts: the order against the checkpoint's own errors and warnings is pinned by `PredictParityTests`. The file's
    // own texts are tested here on the loader.

    private static string Write(string text)
    {
        var path = Path.Combine(Path.GetTempPath(), "laya-calib-" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(path, text);
        return path;
    }

    private static T WithFile<T>(string text, Func<string, T> body)
    {
        var path = Write(text);
        try { return body(path); }
        finally { File.Delete(path); }
    }

    [Fact]
    public void AMalformedFileIsRefusedWithPythonsText()
    {
        var error = Assert.Throws<InvalidDataException>(() => WithFile("[1, 2]", CalibrationFile.Load));
        Assert.Equal("ValueError: calibration JSON must be an object, got list", error.Message);

        error = Assert.Throws<InvalidDataException>(() => WithFile(
            "{\"temperature\": [1.0, 1.0, 1.0], \"binning_map\": {\"choice:3-5\": {\"bins\": 2, \"values\": [0.5, 1.5]}}}", CalibrationFile.Load));
        Assert.Equal(
            "ValueError: calibration JSON binning_map['choice:3-5'] values must be in [0, 1], got [0.5, 1.5]", error.Message);
    }

    [Fact]
    public void AFileThatIsNotJsonOrNotUtf8IsRefusedAsTheConfigIs()
    {
        Assert.ThrowsAny<JsonException>(() => WithFile("{\"temperature\": [1, 2, 3],}", CalibrationFile.Load));
        var path = Write("");
        try
        {
            File.WriteAllBytes(path, [0x7B, 0xFF, 0x7D]);
            Assert.ThrowsAny<JsonException>(() => CalibrationFile.Load(path));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void AMissingFileIsAFileNotFound() =>
        Assert.Throws<FileNotFoundException>(() => CalibrationFile.Load(Path.Combine(Path.GetTempPath(), "absent-" + Guid.NewGuid().ToString("N") + ".json")));

    [Fact]
    public void AVersion2FileFittedForAnotherCheckpointLoadsWithoutAnIdentityWarning()
    {
        // Python's checkpoint-identity warning is not emitted: this SDK has no equivalent of its model_id_or_path string.
        List<string> warnings = [];
        var loaded = CalibrationFile.Apply(PythonJsonReader.Parse(
            "{\"version\": 2, \"temperature\": [1.5, 2.0, 1.25], \"model_id_or_path\": \"somewhere/else\", \"subfolder\": \"x\", \"config\": {\"max_len\": 1}}"),
            warnings.Add);
        Assert.Empty(warnings);
        Assert.Equal([1.5, 2.0, 1.25], loaded.Temperature);
    }

    // ── a router hands the file to every engine it builds ─────────────────────

    [Fact]
    public void ARouterPassesTheFileAndTheLanguageTableToEveryEngine()
    {
        var template = new LayaOptions
        {
            Calibration = "calibration.json",
            LangTemperatures = new Dictionary<string, LayaLangTemperature?> { ["de"] = null },
            ModelDirectory = "ignored",
        };
        foreach (var checkpoint in Enum.GetValues<LayaCheckpoint>())
        {
            var built = LayaRouter.EngineOptions(template, checkpoint);
            Assert.Equal("calibration.json", built.Calibration);
            Assert.Same(template.LangTemperatures, built.LangTemperatures);
            Assert.Equal(checkpoint, built.Checkpoint);
            Assert.Null(built.ModelDirectory);
        }
    }

    // ── the decode of one row ─────────────────────────────────────────────────

    private static readonly IReadOnlyList<double> One = [1.0, 1.0, 1.0];
    private static readonly IReadOnlyDictionary<string, double> NoBuckets = new Dictionary<string, double>();

    private static readonly IReadOnlyDictionary<string, BinningEntry> Binning = new Dictionary<string, BinningEntry>
    {
        ["noul:2"] = new(4, [0.1, 0.4, 0.6, 0.9]),
        ["choice:3-5"] = new(2, [0.25, 0.75]),
    };

    private static Answer Decode(Question question, float[] logits, IReadOnlyDictionary<string, BinningEntry>? binning,
        ResolvedLangTemperature? lang = null, IReadOnlyList<double>? temperature = null,
        IReadOnlyDictionary<string, double>? buckets = null) =>
        LayaEngine.DecodeRow(question, logits, logits.Length, temperature ?? One, buckets ?? NoBuckets, lang, binning, new ActionInfo(0.5));

    private static float[] Logits(double pTrue) => [0f, (float)Math.Log(pTrue / (1 - pTrue))];

    [Fact]
    public void BinsAnswerConfidenceAndNothingElse()
    {
        var question = Question.Noul("Is it?");
        var plain = Decode(question, Logits(0.8), null).AsNoul();
        var binned = Decode(question, Logits(0.8), Binning).AsNoul();

        Assert.Equal(0.8, plain.AnswerConfidence, 4);
        Assert.Equal(0.9, binned.AnswerConfidence);      // 0.8 * 4 = 3.2, bin 3
        Assert.Equal(plain.Probability, binned.Probability);
        Assert.Equal(plain.Confidence, binned.Confidence);
        Assert.Equal(plain.Action, binned.Action);
    }

    [Fact]
    public void ChoiceAndScoreKeepTheirProbabilitiesAndEntropyConfidence()
    {
        float[] logits = [2f, 0.5f, 0f];
        var choice = Question.Choice("Which?", "a", "b", "c");
        var plain = Decode(choice, logits, null).AsChoice();
        var binned = Decode(choice, logits, Binning).AsChoice();
        Assert.Equal(plain.Probabilities, binned.Probabilities);
        Assert.Equal(plain.Confidence, binned.Confidence);
        Assert.Equal(plain.Choice, binned.Choice);
        Assert.Equal(0.75, binned.AnswerConfidence);     // max(p) is about 0.73, bin 1 of 2
        Assert.NotEqual(plain.AnswerConfidence, binned.AnswerConfidence);

        var score = Question.Score("How much?", "low", "mid", "high");
        var plainScore = Decode(score, logits, null).AsScore();
        var binnedScore = Decode(score, logits, Binning).AsScore();   // score:3-5 has no entry: a miss
        Assert.Equal(plainScore.AnswerConfidence, binnedScore.AnswerConfidence);
        Assert.Equal(plainScore.Probabilities, binnedScore.Probabilities);
    }

    [Fact]
    public void BinsTheUnroundedConfidenceAndRoundsTheBinnedValue()
    {
        // max(p) = 0.74996 is in bin 2 of 4; rounded to four places first it would be 0.75, in bin 3.
        var question = Question.Noul("Is it?");
        var answer = Decode(question, Logits(0.74996), Binning).AsNoul();
        Assert.Equal(0.6, answer.AnswerConfidence);
        Assert.Equal(0.75, Calibration.Round4(0.74996));

        // The value read from the map is rounded like any other: 1/3 stays at four places.
        var thirds = new Dictionary<string, BinningEntry> { ["noul:2"] = new(1, [1.0 / 3.0]) };
        Assert.Equal(0.3333, Decode(question, Logits(0.8), thirds).AsNoul().AnswerConfidence);
    }

    [Fact]
    public void AMissingBucketKeepsTheTemperatureScaledValue()
    {
        var question = Question.Noul("Is it?");
        var only = new Dictionary<string, BinningEntry> { ["choice:11+"] = new(2, [0.2, 0.8]) };
        Assert.Equal(Decode(question, Logits(0.8), null).AnswerConfidence, Decode(question, Logits(0.8), only).AnswerConfidence);
    }

    [Fact]
    public void ALanguageTheTableHoldsReplacesTheTemperatureAndSkipsTheBinning()
    {
        var question = Question.Noul("Is it?");
        var lang = new ResolvedLangTemperature([2.0, 2.0, 2.0], new Dictionary<string, double>());
        var plain = Decode(question, Logits(0.8), null, lang);
        var binned = Decode(question, Logits(0.8), Binning, lang);
        Assert.Equal(plain.AnswerConfidence, binned.AnswerConfidence);
        Assert.NotEqual(0.9, binned.AnswerConfidence);
        // The language's temperature (2.0) softened it relative to the file's.
        Assert.True(binned.AnswerConfidence < Decode(question, Logits(0.8), null).AnswerConfidence);
    }

    [Fact]
    public void TheTemperaturesAreTheFilesWhenOneWasLoaded()
    {
        var question = Question.Noul("Is it?");
        var sharp = Decode(question, Logits(0.8), null, temperature: [1.0, 1.0, 0.5]);
        var soft = Decode(question, Logits(0.8), null, temperature: [1.0, 1.0, 2.0]);
        var bucket = Decode(question, Logits(0.8), null, temperature: [1.0, 1.0, 2.0], buckets: new Dictionary<string, double> { ["noul:2"] = 0.5 });
        Assert.True(sharp.AnswerConfidence > soft.AnswerConfidence);
        Assert.Equal(sharp.AnswerConfidence, bucket.AnswerConfidence);
    }

    [Fact]
    public void TheGateReadsTheBinnedValue()
    {
        var question = Question.Noul("Is it?");
        var gate = MinConfidence.Of(0.5).Check();
        // 0.8 passes a 0.5 gate unbinned, but its bin holds 0.4.
        var lowBin = new Dictionary<string, BinningEntry> { ["noul:2"] = new(1, [0.4]) };
        Assert.Equal(GateStates.Passed, ConfidenceGate.Apply(Decode(question, Logits(0.8), null), gate).Abstention);
        Assert.Equal(GateStates.Abstained, ConfidenceGate.Apply(Decode(question, Logits(0.8), lowBin), gate).Abstention);
    }
}
