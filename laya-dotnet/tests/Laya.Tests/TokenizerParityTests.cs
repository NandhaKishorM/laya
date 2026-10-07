using Laya.Tokenization;

namespace Laya.Tests;

/// <summary>
/// Token-level parity against Python for every checkpoint, needing <c>tokenizer.json</c> but not
/// the weights.
/// </summary>
/// <remarks>
/// This is where a port like this most plausibly goes wrong: a divergence of one token shifts every
/// marker after it, and the model then answers a different question without anything failing. These
/// tests compare against ids recorded from the real HuggingFace tokenizer, so they fail loudly
/// instead. Each test receives the checkpoint under test and loads the tokenizer inline, so missing
/// artifacts for one checkpoint skip only that checkpoint's cases.
/// </remarks>
public sealed class TokenizerParityTests
{
    /// <summary>All checkpoints to test against, one theory case per enum value.</summary>
    public static TheoryData<LayaCheckpoint> Checkpoints => new(TestArtifacts.ParityCheckpoints);

    /// <summary>
    /// Cross-product of every checkpoint with every success case in its golden data. Cases for a
    /// checkpoint are omitted entirely when its golden data has not been generated yet.
    /// </summary>
    public static IEnumerable<object[]> CheckpointCases() =>
        from c in TestArtifacts.ParityCheckpoints
        where GoldenData.For(c).Available
        // Kind is null for a standard parity case; see the comment in PredictParityTests.
        from info in GoldenData.For(c).Index.Where(i => i.Kind is null && !i.ExpectsError)
        select new object[] { c, info };

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void SpecialTokenIdsMatchTheRecordedValues(LayaCheckpoint checkpoint)
    {
        var artifacts = TestArtifacts.For(checkpoint);
        var golden = GoldenData.For(checkpoint);
        if (!golden.Available)
            Assert.Skip($"no golden data for '{checkpoint}'; run laya-dotnet/tools/dump_golden.py");

        using var tok = new HfTokenizer(artifacts.RequireTokenizer());
        var expected = golden.Meta.GetProperty("special_tokens");
        Assert.Equal(expected.GetProperty("pad_id").GetInt32(),   tok.PadId);
        Assert.Equal(expected.GetProperty("cls_id").GetInt32(),   tok.ClsId);
        Assert.Equal(expected.GetProperty("sep_id").GetInt32(),   tok.SepId);
        Assert.Equal(expected.GetProperty("mask_id").GetInt32(),  tok.MaskId);
        Assert.Equal(expected.GetProperty("unk_id").GetInt32(),   tok.UnkId);
        Assert.Equal(expected.GetProperty("mask_token").GetString(), tok.MaskToken);
    }

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void EncodesEveryProbeStringExactlyAsPythonDoes(LayaCheckpoint checkpoint)
    {
        var artifacts = TestArtifacts.For(checkpoint);
        var golden = GoldenData.For(checkpoint);
        if (!golden.Available)
            Assert.Skip($"no golden data for '{checkpoint}'; run laya-dotnet/tools/dump_golden.py");

        using var tok = new HfTokenizer(artifacts.RequireTokenizer());
        using var doc = golden.Load("tokenizer_probe.json");
        var failures = new List<string>();

        foreach (var probe in doc.RootElement.EnumerateArray())
        {
            var text = probe.GetProperty("text").GetString()!;
            var expected = GoldenData.Ints(probe.GetProperty("input_ids"));
            var actual = tok.Encode(text);
            if (!expected.SequenceEqual(actual))
                failures.Add($"  {System.Text.Json.JsonSerializer.Serialize(text)}\n" +
                             $"    python: [{string.Join(", ", expected)}]\n" +
                             $"    dotnet: [{string.Join(", ", actual)}]");
        }

        Assert.True(failures.Count == 0,
            $"{failures.Count} probe string(s) tokenized differently:\n{string.Join('\n', failures)}");
    }

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void AddsNoSpecialTokensOfItsOwn(LayaCheckpoint checkpoint)
    {
        // The shipped tokenizer.json has a TemplateProcessing post-processor that wraps every
        // encode. Left in place it would inject the checkpoint's CLS and SEP ids into every
        // fragment, shifting all marker positions. HfTokenizer must strip it.
        var artifacts = TestArtifacts.For(checkpoint);
        using var tok = new HfTokenizer(artifacts.RequireTokenizer());
        var ids = tok.Encode("hello");
        Assert.NotEqual(tok.ClsId, ids[0]);
        Assert.NotEqual(tok.SepId, ids[^1]);
    }

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void EncodesTheEmptyStringToNothing(LayaCheckpoint checkpoint)
    {
        var artifacts = TestArtifacts.For(checkpoint);
        using var tok = new HfTokenizer(artifacts.RequireTokenizer());
        Assert.Empty(tok.Encode(""));
    }

    [Theory]
    [MemberData(nameof(CheckpointCases))]
    public void BuildsTheRecordedSequenceForEveryQuestion(LayaCheckpoint checkpoint, GoldenCaseInfo info)
    {
        var artifacts = TestArtifacts.For(checkpoint);
        var golden    = GoldenData.For(checkpoint);
        using var tok = new HfTokenizer(artifacts.RequireTokenizer());
        using var doc = golden.Load(info);
        var root      = doc.RootElement;
        var state     = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        var maxLen     = golden.Meta.GetProperty("max_len").GetInt32();
        var headMaxLen = golden.Meta.GetProperty("head_max_len").GetInt32();

        foreach (var recorded in root.GetProperty("per_question").EnumerateArray())
        {
            var id       = recorded.GetProperty("id").GetString()!;
            var question = questions[id];

            // The rendered option strings come first: they are the readable form of the divergence,
            // so a mismatch here explains a mismatch in the ids below.
            Assert.Equal([.. recorded.GetProperty("options").EnumerateArray().Select(e => e.GetString()!)],
                         SequenceBuilder.RenderOptions(question));
            Assert.Equal(recorded.GetProperty("instructions").GetString(),
                         SequenceBuilder.RenderInstructions(question));

            var (ids, markers) = SequenceBuilder.Build(tok, state, question, maxLen, headMaxLen);
            Assert.Equal(GoldenData.Ints(recorded.GetProperty("markers")), markers);
            Assert.Equal(GoldenData.Ints(recorded.GetProperty("input_ids")), ids);
        }
    }

    [Theory]
    [MemberData(nameof(CheckpointCases))]
    public void SerializesStateExactlyAsPythonDoes(LayaCheckpoint checkpoint, GoldenCaseInfo info)
    {
        // State serialization is checkpoint-independent (it is the same JSON serializer for all),
        // but exercising it across checkpoints gives coverage over diverse state shapes in the
        // golden corpus of every checkpoint.
        _ = TestArtifacts.For(checkpoint); // not needed here, but keeps the method signature consistent
        var golden = GoldenData.For(checkpoint);
        using var doc = golden.Load(info);
        var root  = doc.RootElement;
        var state = GoldenData.ToClr(root.GetProperty("state"));
        Assert.Equal(root.GetProperty("serialized_state").GetString(), PythonJson.State(state));
    }
}
