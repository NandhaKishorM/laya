namespace Laya;

/// <summary>One question, tokenized and ready to be batched.</summary>
/// <param name="Ids">The token sequence.</param>
/// <param name="Markers">One position per option, pointing at that option's <c>[MASK]</c>.</param>
/// <param name="Type">Which question type, fed to the model's type embedding.</param>
public readonly record struct SequenceItem(int[] Ids, int[] Markers, QuestionType Type);

/// <summary>A padded batch, laid out as the row-major tensors the ONNX graph takes.</summary>
public sealed class CollatedBatch
{
    internal CollatedBatch(int count, int sequenceLength, int markerCount,
                           long[] inputIds, long[] attentionMask, long[] markerPos,
                           bool[] markerMask, long[] qtype, int[] markerCounts, int inputTokens)
    {
        Count = count;
        SequenceLength = sequenceLength;
        MarkerCount = markerCount;
        InputIds = inputIds;
        AttentionMask = attentionMask;
        MarkerPos = markerPos;
        MarkerMask = markerMask;
        QType = qtype;
        MarkerCounts = markerCounts;
        InputTokens = inputTokens;
    }

    /// <summary>Rows in the batch: one per question.</summary>
    public int Count { get; }

    /// <summary>Padded sequence length, the longest row.</summary>
    public int SequenceLength { get; }

    /// <summary>Padded marker-axis width.</summary>
    public int MarkerCount { get; }

    /// <summary><c>[Count, SequenceLength]</c>, padded with the pad id.</summary>
    public long[] InputIds { get; }

    /// <summary><c>[Count, SequenceLength]</c>, 1 on real tokens and 0 on padding.</summary>
    public long[] AttentionMask { get; }

    /// <summary><c>[Count, MarkerCount]</c>, 0 in padding columns.</summary>
    public long[] MarkerPos { get; }

    /// <summary><c>[Count, MarkerCount]</c>, false in padding columns.</summary>
    public bool[] MarkerMask { get; }

    /// <summary><c>[Count]</c>, the question type per row.</summary>
    public long[] QType { get; }

    /// <summary>The real (unpadded) option count per row — how many logits to read from that row.</summary>
    public int[] MarkerCounts { get; }

    /// <summary>Non-padding tokens across the batch, which is what Python reports as input tokens.</summary>
    public int InputTokens { get; }
}

/// <summary>Pads tokenized questions into one batch. Ports <c>collate_items</c> from <c>laya/common.py</c>.</summary>
public static class Collator
{
    /// <summary>
    /// The marker axis is never narrower than this.
    /// </summary>
    /// <remarks>
    /// The tracer baked <c>TopK k=2</c> into the exported graph: Python chooses the top-2 branch at
    /// runtime and the export only ever saw two or more markers, so a batch whose questions all
    /// have a single option makes TopK fail outright. Padding to two columns with
    /// <c>marker_mask = false</c> is exactly equivalent, not merely close: the model's
    /// <c>masked_fill</c> puts <c>-1e4</c> in the pad column, which underflows softmax to 0.0, so
    /// <c>p == [1.0, 0.0]</c> — precisely the <c>[top1, 0.0]</c> Python substitutes — and
    /// <c>k = marker_mask.sum().clamp(min=2) == 2</c> either way. Reading only
    /// <see cref="CollatedBatch.MarkerCounts"/> logits per row keeps the pad column out of the answer.
    /// </remarks>
    public const int MinMarkers = 2;

    /// <summary>Pad a batch of tokenized questions.</summary>
    public static CollatedBatch Collate(IReadOnlyList<SequenceItem> items, int padId)
    {
        ArgumentNullException.ThrowIfNull(items);
        if (items.Count == 0) throw new ArgumentException("nothing to collate", nameof(items));

        var n = items.Count;
        var length = items.Max(it => it.Ids.Length);
        var markerCount = Math.Max(MinMarkers, items.Max(it => it.Markers.Length));

        var ids = new long[n * length];
        if (padId != 0) Array.Fill(ids, padId);
        var attention = new long[n * length];
        var markerPos = new long[n * markerCount];
        var markerMask = new bool[n * markerCount];
        var qtype = new long[n];
        var markerCounts = new int[n];
        var inputTokens = 0;

        for (var i = 0; i < n; i++)
        {
            var item = items[i];
            var row = i * length;
            for (var j = 0; j < item.Ids.Length; j++)
            {
                ids[row + j] = item.Ids[j];
                attention[row + j] = 1;
            }
            inputTokens += item.Ids.Length;

            var mrow = i * markerCount;
            for (var j = 0; j < item.Markers.Length; j++)
            {
                markerPos[mrow + j] = item.Markers[j];
                markerMask[mrow + j] = true;
            }

            qtype[i] = (long)item.Type;
            markerCounts[i] = item.Markers.Length;
        }

        return new CollatedBatch(n, length, markerCount, ids, attention, markerPos, markerMask,
                                 qtype, markerCounts, inputTokens);
    }
}
