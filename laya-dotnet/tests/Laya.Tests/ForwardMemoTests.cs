namespace Laya.Tests;

/// <summary>Model-free: the forward-pass memo answers only byte-identical inputs from its store.</summary>
public sealed class ForwardMemoTests
{
    private static CollatedBatch Batch(params SequenceItem[] items) => Collator.Collate(items, padId: 0);

    private static SequenceItem Item(int[] ids, int[] markers, QuestionType type = QuestionType.Choice) =>
        new(ids, markers, type);

    private static SequenceItem Base() => Item([5, 6, 7, 8], [1, 2]);

    /// <summary>Counts the real runs and returns outputs derived from the call count.</summary>
    private sealed class Graph
    {
        public int Calls;

        public (float[] Logits, float[] ActLogits) Run(CollatedBatch batch)
        {
            Calls++;
            return ([batch.InputIds.Sum() + Calls, 1f], [batch.MarkerPos.Sum(), 2f]);
        }
    }

    [Fact]
    public void AnIdenticalBatchRunsTheGraphOnce()
    {
        var memo = new ForwardMemo();
        var graph = new Graph();
        var first = memo.Forward(Batch(Base()), graph.Run);
        var second = memo.Forward(Batch(Base()), graph.Run);

        Assert.Equal(1, graph.Calls);
        Assert.Equal((1, 1), (memo.Runs, memo.Hits));
        Assert.Equal(first.Logits, second.Logits);
        Assert.Equal(first.ActLogits, second.ActLogits);
    }

    [Fact]
    public void AHitReturnsCopiesSoACallerCannotAlterTheStore()
    {
        var memo = new ForwardMemo();
        var graph = new Graph();
        var first = memo.Forward(Batch(Base()), graph.Run);
        var expected = (float[])first.Logits.Clone();
        first.Logits[0] = -1f;
        var second = memo.Forward(Batch(Base()), graph.Run);
        second.Logits[0] = -2f;
        var third = memo.Forward(Batch(Base()), graph.Run);

        Assert.Equal(expected, third.Logits);
    }

    public static IEnumerable<object[]> DifferingBatches()
    {
        yield return [Batch(Item([5, 6, 7, 9], [1, 2]))];                    // one token id
        yield return [Batch(Item([5, 6, 7, 8], [1, 3]))];                    // one marker position
        yield return [Batch(Item([5, 6, 7, 8], [1, 2], QuestionType.Noul))]; // qtype
        yield return [Batch(Item([5, 6, 7, 8], [1, 2, 3]))];                 // marker count (shape and mask)
        yield return [Batch(Item([5, 6, 7, 8, 8], [1, 2]))];                 // sequence length
        yield return [Batch(Base(), Base())];                                // row count
        yield return [Batch(Base(), Item([5, 6], [1, 0]))];                  // padding and attention mask differ
    }

    [Theory]
    [MemberData(nameof(DifferingBatches))]
    public void AnyDifferingInputRunsTheGraphAgain(CollatedBatch other)
    {
        var memo = new ForwardMemo();
        var graph = new Graph();
        memo.Forward(Batch(Base()), graph.Run);
        memo.Forward(other, graph.Run);

        Assert.Equal(2, graph.Calls);
        Assert.Equal(0, memo.Hits);
    }

    [Fact]
    public void ADifferingAttentionMaskIsAMissEvenWhenTheShapeMatches()
    {
        // Same shape: the second row is shorter, so its padding and attention mask differ.
        var memo = new ForwardMemo();
        var graph = new Graph();
        memo.Forward(Batch(Item([5, 6, 7, 8], [1, 2]), Item([5, 6, 7, 8], [1, 2])), graph.Run);
        memo.Forward(Batch(Item([5, 6, 7, 8], [1, 2]), Item([5, 6, 7, 8], [1, 2])), graph.Run);
        Assert.Equal(1, graph.Calls);
        memo.Forward(Batch(Item([5, 6, 7, 8], [1, 2]), Item([5, 6, 7], [1, 2])), graph.Run);
        Assert.Equal(2, graph.Calls);
    }

    [Fact]
    public void CheckpointsAndLayoutsKeepSeparateMemos()
    {
        var batch = Batch(Base());
        var graph = new Graph();
        var memos = (from checkpoint in Enum.GetValues<LayaCheckpoint>()
                     from split in new[] { false, true }
                     select ForwardMemos.For(checkpoint, split)).ToArray();

        Assert.Equal(memos.Length, memos.Distinct().Count());
        Assert.Same(memos[0], ForwardMemos.For(Enum.GetValues<LayaCheckpoint>()[0], false));
        // The same batch in every memo reaches the graph every time: nothing leaks across memos.
        foreach (var memo in memos) memo.Forward(batch, graph.Run);
        Assert.Equal(memos.Length, graph.Calls);
    }
}
