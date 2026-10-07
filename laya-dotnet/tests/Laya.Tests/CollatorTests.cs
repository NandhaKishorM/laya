namespace Laya.Tests;

/// <summary>Tier 1: batch padding, which needs no artifacts.</summary>
public sealed class CollatorTests
{
    private static SequenceItem Item(int length, int markers, QuestionType type = QuestionType.Choice) =>
        new([.. Enumerable.Range(10, length)], [.. Enumerable.Range(1, markers)], type);

    [Fact]
    public void PadsToTheLongestRowAndMarksAttentionAccordingly()
    {
        var batch = Collator.Collate([Item(5, 2), Item(3, 2)], padId: 0);

        Assert.Equal(2, batch.Count);
        Assert.Equal(5, batch.SequenceLength);
        Assert.Equal([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], batch.AttentionMask);
        // The short row's tail is the pad id, and attention is 0 there.
        Assert.Equal(0, batch.InputIds[8]);
        Assert.Equal(0, batch.InputIds[9]);
    }

    [Fact]
    public void FillsWithTheGivenPadIdRatherThanZero()
    {
        var batch = Collator.Collate([Item(3, 2), Item(1, 2)], padId: 77);
        Assert.Equal(77, batch.InputIds[4]);
        Assert.Equal(77, batch.InputIds[5]);
        Assert.Equal(0, batch.AttentionMask[4]);
    }

    [Fact]
    public void CountsOnlyRealTokensAsInputTokens()
    {
        var batch = Collator.Collate([Item(5, 2), Item(3, 2)], padId: 0);
        Assert.Equal(8, batch.InputTokens);
    }

    [Fact]
    public void WidensTheMarkerAxisToTwoForSingleOptionBatches()
    {
        // The exported graph has TopK k=2 baked in: a one-column marker axis makes it fail outright.
        var batch = Collator.Collate([Item(6, 1)], padId: 0);

        Assert.Equal(Collator.MinMarkers, batch.MarkerCount);
        Assert.Equal([1, 0], batch.MarkerPos);
        Assert.Equal([true, false], batch.MarkerMask);
        // The real option count is kept separately, so the pad column never reaches an answer.
        Assert.Equal([1], batch.MarkerCounts);
    }

    [Fact]
    public void PadsTheMarkerAxisToTheWidestQuestion()
    {
        var batch = Collator.Collate([Item(6, 4), Item(6, 2)], padId: 0);

        Assert.Equal(4, batch.MarkerCount);
        Assert.Equal([1, 2, 3, 4, 1, 2, 0, 0], batch.MarkerPos);
        Assert.Equal([true, true, true, true, true, true, false, false], batch.MarkerMask);
        Assert.Equal([4, 2], batch.MarkerCounts);
    }

    [Fact]
    public void CarriesTheQuestionTypePerRowUsingPythonsNumbering()
    {
        var batch = Collator.Collate(
            [Item(4, 2, QuestionType.Noul), Item(4, 2, QuestionType.Choice), Item(4, 2, QuestionType.Score)],
            padId: 0);

        Assert.Equal([2, 0, 1], batch.QType);
    }

    [Fact]
    public void RefusesAnEmptyBatch() =>
        Assert.Throws<ArgumentException>(() => Collator.Collate([], padId: 0));

    [Theory]
    [MemberData(nameof(GoldenData.SuccessCases), MemberType = typeof(GoldenData))]
    public void ReproducesTheRecordedBatchTensors(GoldenCaseInfo info)
    {
        using var doc = GoldenData.Load(info);
        var root = doc.RootElement;
        var expected = root.GetProperty("batch");

        // Rebuild the batch from the recorded per-question ids, so this checks the collator alone
        // and stays meaningful without a tokenizer present.
        var items = new List<SequenceItem>();
        foreach (var q in root.GetProperty("per_question").EnumerateArray())
            items.Add(new SequenceItem(
                GoldenData.Ints(q.GetProperty("input_ids")),
                GoldenData.Ints(q.GetProperty("markers")),
                (QuestionType)q.GetProperty("qtype").GetInt32()));

        var batch = Collator.Collate(items, padId: GoldenData.Meta.GetProperty("special_tokens")
                                                                 .GetProperty("pad_id").GetInt32());

        AssertMatrix(expected.GetProperty("input_ids"), batch.InputIds, batch.SequenceLength);
        AssertMatrix(expected.GetProperty("attention_mask"), batch.AttentionMask, batch.SequenceLength);
        AssertMatrix(expected.GetProperty("marker_pos"), batch.MarkerPos, batch.MarkerCount);
        Assert.Equal([.. expected.GetProperty("qtype").EnumerateArray().Select(e => e.GetInt64())], batch.QType);

        var expectedMask = expected.GetProperty("marker_mask");
        Assert.Equal(expectedMask.GetArrayLength(), batch.Count);
        for (var r = 0; r < batch.Count; r++)
        {
            var row = expectedMask[r];
            for (var c = 0; c < batch.MarkerCount; c++)
                Assert.Equal(row[c].GetBoolean(), batch.MarkerMask[r * batch.MarkerCount + c]);
        }

        Assert.Equal(root.GetProperty("result").GetProperty("usage").GetProperty("input_tokens").GetInt32(),
                     batch.InputTokens);
        // marker_pos above already pinned the padded width; this checks the flag the dump script
        // set, so a future golden that stops exercising the TopK edge case is noticed.
        Assert.Equal(root.GetProperty("marker_axis_padded").GetBoolean(),
                     batch.MarkerCount > items.Max(i => i.Markers.Length));
    }

    private static void AssertMatrix(System.Text.Json.JsonElement expected, long[] actual, int cols)
    {
        var rows = expected.GetArrayLength();
        Assert.Equal(rows * cols, actual.Length);
        for (var r = 0; r < rows; r++)
        {
            var row = expected[r];
            Assert.Equal(cols, row.GetArrayLength());
            for (var c = 0; c < cols; c++)
                Assert.Equal(row[c].GetInt64(), actual[r * cols + c]);
        }
    }
}
