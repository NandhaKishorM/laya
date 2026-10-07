namespace Laya.Tests;

/// <summary>
/// Tier 2: the exported graph's dynamic axes and its raw outputs, driven from C#.
/// </summary>
/// <remarks>
/// The export was traced at one shape, so every axis being genuinely dynamic is a property of the
/// artifact rather than something the exporter guarantees. This sweep re-runs from .NET the 72
/// configurations the Python probe covered, and then checks the graph outputs against recorded ones,
/// so a bad artifact is caught here rather than as a puzzling answer mismatch elsewhere.
/// </remarks>
[Collection(AllEnginesCollection.Name)]
public sealed class ShapeSweepTests(AllEnginesFixture fixture)
{
    /// <summary>
    /// Candidate sequence lengths, straddling the multilingual traced length of 93.
    /// Each checkpoint's actual sweep is this list filtered to <c>&lt;= Config.MaxLen</c>,
    /// with <c>MaxLen</c> always included. The English checkpoint caps at 512 so it gets
    /// 11 of these lengths; typed-decisions and multilingual (both 1024) get all 12.
    /// </summary>
    public static readonly int[] SequenceLengths = [16, 32, 55, 59, 80, 92, 93, 94, 128, 256, 512, 1024];

    /// <summary>Batch sizes: one row, and more than one.</summary>
    public static readonly int[] BatchSizes = [1, 3];

    /// <summary>Marker counts, from the graph's TopK floor of 2 upwards.</summary>
    public static readonly int[] MarkerCounts = [2, 5, 9];

    /// <summary>All 72 multilingual shape combinations.</summary>
    public static IEnumerable<object[]> Shapes() =>
        from seq in SequenceLengths
        from batch in BatchSizes
        from markers in MarkerCounts
        select new object[] { seq, batch, markers };

    /// <summary>
    /// Shape combinations for all checkpoints. For each checkpoint the sequence lengths are
    /// <see cref="SequenceLengths"/> filtered to at most <c>Config.MaxLen</c>, with
    /// <c>MaxLen</c> itself always included. The limit is read from the artifact config at
    /// theory-data time; when the artifact is absent the full list is used and the test will
    /// skip when <see cref="AllEnginesFixture.Require"/> is called.
    /// </summary>
    public static IEnumerable<object[]> CheckpointShapes()
    {
        foreach (var checkpoint in TestArtifacts.ParityCheckpoints)
        {
            // Derive max_len from the artifact config without loading the ONNX graph.
            // Default to 1024 (the largest value across all checkpoints) when the artifact
            // directory cannot be found; those cases will be skipped at run time.
            var maxLen = 1024;
            var dir = TestArtifacts.For(checkpoint).DirectoryOrNull;
            if (dir is not null)
            {
                try
                {
                    maxLen = LayaConfig.Load(
                        System.IO.Path.Combine(dir, "rl_agent_config.json")).MaxLen;
                }
                catch
                {
                    // Config unreadable; fall back to 1024 and let fixture.Require() skip.
                }
            }

            var seqs = SequenceLengths.Where(s => s <= maxLen).ToList();
            if (!seqs.Contains(maxLen)) seqs.Add(maxLen);
            seqs.Sort();

            foreach (var seq in seqs)
                foreach (var batch in BatchSizes)
                    foreach (var markers in MarkerCounts)
                        yield return [checkpoint, seq, batch, markers];
        }
    }

    // ── multilingual-only shape tests ─────────────────────────────────────────

    [Theory]
    [MemberData(nameof(Shapes))]
    public void RunsAtEveryShape(int seq, int batch, int markers)
    {
        var engine   = fixture.Require(LayaCheckpoint.Multilingual);
        var collated = Synthesize(seq, batch, markers);
        var (logits, actLogits) = engine.Run(collated);

        Assert.Equal(batch * markers, logits.Length);
        Assert.Equal(batch * 2, actLogits.Length);
        // NaN here would not throw but would poison every probability downstream into silence.
        Assert.All(logits, v => Assert.True(float.IsFinite(v), $"non-finite logit at seq={seq}, markers={markers}"));
        Assert.All(actLogits, v => Assert.True(float.IsFinite(v), $"non-finite act logit at seq={seq}"));
    }

    [Theory]
    [MemberData(nameof(GoldenData.SuccessCases), MemberType = typeof(GoldenData))]
    public void ReproducesTheRecordedGraphOutputs(GoldenCaseInfo info)
    {
        var engine = fixture.Require(LayaCheckpoint.Multilingual);
        using var doc = GoldenData.Load(info);
        var root = doc.RootElement;

        // Feed the tensors Python recorded rather than ones rebuilt here, so this isolates the ONNX
        // run from tokenization: a mismatch is the graph or the tensor marshalling, nothing else.
        var batch = FromRecorded(root.GetProperty("batch"));
        var (logits, actLogits) = engine.Run(batch);

        var expectedLogits = GoldenData.Matrix(root.GetProperty("logits"), out var rows, out var cols);
        Assert.Equal(batch.Count, rows);
        Assert.Equal(batch.MarkerCount, cols);
        // Logits live around ±10 (and exactly -1e4 in padded columns), so this is a tight bound.
        AssertClose(expectedLogits, logits, 2e-3, "logits");

        // act_logits reach ±1600, where float32 spacing alone is ~1e-4; 0.05 is ~3e-5 relative.
        var expectedAct = GoldenData.Matrix(root.GetProperty("act_logits"), out _, out _);
        AssertClose(expectedAct, actLogits, 0.05, "act_logits");
    }

    [Fact]
    public void KeepsPaddedMarkerColumnsAtTheMaskedFillValue()
    {
        // The model writes -1e4 into columns marker_mask says are padding. Calibration relies on that
        // to keep a padded column out of a distribution, so it is worth pinning on the graph itself.
        var engine = fixture.Require(LayaCheckpoint.Multilingual);
        var items = new List<SequenceItem>
        {
            Item(seq: 32, markers: 4),
            Item(seq: 32, markers: 2),
        };
        var batch = Collator.Collate(items, padId: 0);
        var (logits, _) = engine.Run(batch);

        Assert.Equal(4, batch.MarkerCount);
        Assert.True(logits[1 * 4 + 2] < -1e3, $"padded column carried {logits[1 * 4 + 2]}");
        Assert.True(logits[1 * 4 + 3] < -1e3, $"padded column carried {logits[1 * 4 + 3]}");
        Assert.True(float.IsFinite(logits[1 * 4 + 1]));
    }

    [Fact]
    public void RunsASingleOptionQuestionThatTopKWouldOtherwiseReject()
    {
        // The graph has TopK k=2 baked in, so a one-column marker axis fails outright. Collator widens
        // it to two; this is the test that the widening is what actually keeps the graph runnable.
        var engine = fixture.Require(LayaCheckpoint.Multilingual);
        var batch  = Collator.Collate([Item(seq: 24, markers: 1)], padId: 0);

        Assert.Equal(2, batch.MarkerCount);
        var (logits, _) = engine.Run(batch);
        Assert.True(float.IsFinite(logits[0]));
    }

    // ── all-checkpoint shape test ─────────────────────────────────────────────

    [Theory]
    [MemberData(nameof(CheckpointShapes))]
    public void RunsAtEveryShapeForAllCheckpoints(LayaCheckpoint checkpoint, int seq, int batch, int markers)
    {
        var engine   = fixture.Require(checkpoint);
        var collated = Synthesize(seq, batch, markers);
        var (logits, actLogits) = engine.Run(collated);

        Assert.Equal(batch * markers, logits.Length);
        Assert.Equal(batch * 2, actLogits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v),
            $"non-finite logit: checkpoint={checkpoint}, seq={seq}, markers={markers}"));
        Assert.All(actLogits, v => Assert.True(float.IsFinite(v),
            $"non-finite act logit: checkpoint={checkpoint}, seq={seq}"));
    }

    // ── split-layout shape test ────────────────────────────────────────────────

    /// <summary>
    /// Same shape combinations as <see cref="CheckpointShapes"/>, but reading each checkpoint's
    /// <c>MaxLen</c> from the split artifact directory (<c>&lt;repo&gt;/onnx-split/&lt;checkpoint&gt;</c>)
    /// when present, since a split export could in principle carry a different
    /// <c>rl_agent_config.json</c> than its fused sibling.
    /// </summary>
    public static IEnumerable<object[]> CheckpointShapesSplit()
    {
        foreach (var checkpoint in TestArtifacts.ParityCheckpoints)
        {
            var maxLen = 1024;
            var dir = TestArtifacts.For(checkpoint).SplitDirectoryOrNull;
            if (dir is not null)
            {
                try
                {
                    maxLen = LayaConfig.Load(
                        System.IO.Path.Combine(dir, "rl_agent_config.json")).MaxLen;
                }
                catch
                {
                    // Config unreadable; fall back to 1024 and let fixture.RequireSplit() skip.
                }
            }

            var seqs = SequenceLengths.Where(s => s <= maxLen).ToList();
            if (!seqs.Contains(maxLen)) seqs.Add(maxLen);
            seqs.Sort();

            foreach (var seq in seqs)
                foreach (var batch in BatchSizes)
                    foreach (var markers in MarkerCounts)
                        yield return [checkpoint, seq, batch, markers];
        }
    }

    /// <summary>
    /// Same assertions as <see cref="RunsAtEveryShapeForAllCheckpoints"/>, run against the
    /// split-layout engine (<c>encoder.onnx</c> + <c>head.onnx</c>). <see cref="LayaEngine.Run"/>
    /// already dispatches to the split path internally, so this is the same plumbing under a
    /// different artifact directory rather than a separate code path being exercised here.
    /// </summary>
    [Theory]
    [MemberData(nameof(CheckpointShapesSplit))]
    public void RunsAtEveryShapeForAllCheckpointsSplit(LayaCheckpoint checkpoint, int seq, int batch, int markers)
    {
        var engine   = fixture.RequireSplit(checkpoint);
        var collated = Synthesize(seq, batch, markers);
        var (logits, actLogits) = engine.Run(collated);

        Assert.Equal(batch * markers, logits.Length);
        Assert.Equal(batch * 2, actLogits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v),
            $"non-finite logit: checkpoint={checkpoint}, seq={seq}, markers={markers}"));
        Assert.All(actLogits, v => Assert.True(float.IsFinite(v),
            $"non-finite act logit: checkpoint={checkpoint}, seq={seq}"));
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static SequenceItem Item(int seq, int markers, QuestionType type = QuestionType.Choice)
    {
        // A plausible sequence: CLS, ordinary vocabulary ids, SEP. The ids only have to be in range —
        // this exercises shapes, not meaning.
        var ids = new int[seq];
        ids[0] = 2;
        for (var i = 1; i < seq - 1; i++) ids[i] = 100 + (i % 5000);
        ids[seq - 1] = 1;

        // Markers sit just after CLS and stay inside the sequence.
        var positions = new int[markers];
        for (var i = 0; i < markers; i++) positions[i] = 1 + i;
        return new SequenceItem(ids, positions, type);
    }

    private static CollatedBatch Synthesize(int seq, int batch, int markers)
    {
        var types = new[] { QuestionType.Choice, QuestionType.Score, QuestionType.Noul };
        var items = new List<SequenceItem>(batch);
        for (var r = 0; r < batch; r++) items.Add(Item(seq, markers, types[r % types.Length]));
        return Collator.Collate(items, padId: 0);
    }

    private static CollatedBatch FromRecorded(System.Text.Json.JsonElement recorded)
    {
        // Reconstruct items from the recorded rows, dropping the padding the collator will re-add.
        var attention = recorded.GetProperty("attention_mask");
        var inputIds  = recorded.GetProperty("input_ids");
        var markerPos = recorded.GetProperty("marker_pos");
        var markerMask = recorded.GetProperty("marker_mask");
        var qtype     = recorded.GetProperty("qtype");

        var items = new List<SequenceItem>(inputIds.GetArrayLength());
        for (var r = 0; r < inputIds.GetArrayLength(); r++)
        {
            var length  = attention[r].EnumerateArray().Count(e => e.GetInt32() != 0);
            var ids     = GoldenData.Ints(inputIds[r]).Take(length).ToArray();
            var markers = GoldenData.Ints(markerPos[r])
                .Where((_, i) => markerMask[r][i].GetBoolean())
                .ToArray();
            items.Add(new SequenceItem(ids, markers, (QuestionType)qtype[r].GetInt32()));
        }

        return Collator.Collate(items, padId: GoldenData.Meta.GetProperty("special_tokens")
                                                             .GetProperty("pad_id").GetInt32());
    }

    private static void AssertClose(float[] expected, float[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        var worst = 0.0;
        // Index 0, not -1: the failure message is built eagerly, so an all-exact comparison must not
        // index out of range on its way to passing.
        var at = 0;
        for (var i = 0; i < expected.Length; i++)
        {
            var diff = Math.Abs((double)expected[i] - actual[i]);
            if (diff > worst) (worst, at) = (diff, i);
        }
        Assert.True(worst <= tolerance,
            $"{what}: worst difference {worst} at index {at} (python {expected[at]}, dotnet {actual[at]}), " +
            $"tolerance {tolerance}");
    }
}
