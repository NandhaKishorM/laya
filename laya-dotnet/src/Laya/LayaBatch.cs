namespace Laya;

/// <summary>
/// Per-call knobs for <see cref="ILayaBatchPredictor.PredictBatch"/> and
/// <see cref="LayaRouter.PredictBatch"/>. Ports the batching arguments of Python's
/// <c>ONNXAgent.predict_batch</c> (<c>batch_size</c>, <c>sort_by_length</c>, <c>min_confidence</c>, <c>lang</c>,
/// <c>max_len</c>, <c>head_max_len</c>). It is a class with init-only properties so further per-call options can be
/// added without changing any signature. A router takes <c>lang</c>, <c>max_len</c> and <c>head_max_len</c> per request
/// (<see cref="LayaBatchRequest"/>), as Python's <c>Router.predict_batch</c> does, and refuses them here.
/// </summary>
public sealed class LayaBatchOptions
{
    /// <summary>
    /// Cap on states per session run. <see langword="null"/>, zero or a negative value sends them all
    /// in one run, exactly as Python's <c>batch_size</c> (<c>batch_size if batch_size and batch_size &gt; 0
    /// else len(states)</c>); Python raises nothing for such values.
    /// </summary>
    public int? BatchSize { get; init; }

    /// <summary>
    /// Group similarly sized encoded states within windows of eight batches to cut padding. Takes
    /// effect only with an explicit <see cref="BatchSize"/> greater than one and smaller than the
    /// number of states (per question group on a router); otherwise it is ignored. Results keep
    /// input order. Changing batch shapes can slightly change floating-point predictions near
    /// decision thresholds.
    /// </summary>
    public bool SortByLength { get; init; }

    /// <summary>
    /// Opt-in abstention threshold (Python <c>min_confidence</c>), applied to every answer of every state
    /// in the call; on a router it is batch-wide, not part of the request grouping. <see langword="null"/>
    /// writes nothing onto the answers. See <see cref="Laya.MinConfidence"/>.
    /// </summary>
    public MinConfidence? MinConfidence { get; init; }

    /// <summary>
    /// Language of every state in the call (Python <c>lang</c>); see <see cref="LayaPredictOptions.Lang"/>.
    /// </summary>
    public string? Lang { get; init; }

    /// <summary>
    /// Total sequence budget for every state in the call (Python <c>max_len</c>); see
    /// <see cref="LayaPredictOptions.MaxLen"/>.
    /// </summary>
    public int? MaxLen { get; init; }

    /// <summary>
    /// Question-head budget for every state in the call (Python <c>head_max_len</c>); see
    /// <see cref="LayaPredictOptions.HeadMaxLen"/>.
    /// </summary>
    public int? HeadMaxLen { get; init; }
}

/// <summary>
/// Anything that can answer one <see cref="QuestionSet"/> over many states in shared forward passes:
/// a <see cref="LayaEngine"/>. <see cref="LayaRouter.PredictBatch"/> needs its checkpoints to
/// implement this. A separate interface, so <see cref="ILayaPredictor"/> keeps its shape.
/// </summary>
public interface ILayaBatchPredictor : ILayaPredictor
{
    /// <summary>
    /// Evaluate the same <paramref name="questions"/> over every state, one result per state in input order.
    /// </summary>
    IReadOnlyList<LayaResult> PredictBatch(
        IReadOnlyList<object?> states, QuestionSet questions, LayaBatchOptions? options = null);
}

/// <summary>
/// One request of a heterogeneous <see cref="LayaRouter.PredictBatch"/> / <see cref="LayaRouter.RouteBatch"/>
/// call. Python passes these as dicts with <c>state</c> and <c>questions</c> plus the optional routing
/// overrides <c>model</c>, <c>task</c>, <c>lang</c> and <c>lang_guess</c>, and the token-budget overrides
/// <c>max_len</c> and <c>head_max_len</c>.
/// </summary>
public sealed class LayaBatchRequest
{
    /// <summary>The state to judge; null is refused, as Python refuses a <c>None</c> state.</summary>
    public required object? State { get; init; }

    /// <summary>The questions to answer.</summary>
    public required QuestionSet Questions { get; init; }

    /// <summary>Explicit checkpoint name or alias (Python <c>model</c>).</summary>
    public string? Model { get; init; }

    /// <summary>Explicit task name (Python <c>task</c>).</summary>
    public string? Task { get; init; }

    /// <summary>
    /// Explicit language code (Python <c>lang</c>): routes the request and, on a checkpoint that has
    /// <see cref="LayaOptions.LangTemperatures"/>, selects the language's temperatures. Without it the router's own detection
    /// of the state's language is used for the latter.
    /// </summary>
    public string? Lang { get; init; }

    /// <summary>Per-request language hint (Python <c>lang_guess</c>).</summary>
    public LangGuess? LangGuess { get; init; }

    /// <summary>
    /// Total sequence budget for this request (Python <c>max_len</c>); see <see cref="LayaPredictOptions.MaxLen"/>.
    /// Requests that ask for different budgets are answered in separate forward passes.
    /// </summary>
    public int? MaxLen { get; init; }

    /// <summary>
    /// Question-head budget for this request (Python <c>head_max_len</c>); see <see cref="LayaPredictOptions.HeadMaxLen"/>.
    /// </summary>
    public int? HeadMaxLen { get; init; }
}

/// <summary>Python's refusal of a missing state, shared by every prediction entry point.</summary>
internal static class StateGuard
{
    /// <summary>The text of Python's <c>TypeError</c> (onnx_agent.py:413, router.py:1335).</summary>
    internal const string NullStateText = "state must not be None; pass a string, dict, or list";

    /// <summary>Refuse a null state: Python raises instead of answering a decision about the text <c>"null"</c>.</summary>
    internal static void NotNull(object? state)
    {
        if (state is null) throw new ArgumentNullException(null, NullStateText);
    }
}

/// <summary>Chunking and ordering rules of Python's <c>ONNXAgent._infer_batch</c>, free of any model.</summary>
internal static class BatchPlanning
{
    /// <summary>Number of windows of batches the sort looks ahead over (onnx_agent.py:815).</summary>
    internal const int LookaheadBatches = 8;

    /// <summary>
    /// <c>chunk</c>, <c>reorder</c> and <c>window</c> of onnx_agent.py:809-815. <paramref name="count"/> must be positive.
    /// </summary>
    internal static (int Chunk, bool Reorder, int Window) Plan(int count, int? batchSize, bool sortByLength)
    {
        var chunk = batchSize is > 0 ? batchSize.Value : count;
        var reorder = sortByLength && 1 < chunk && chunk < count;
        var window = reorder ? (int)Math.Min((long)chunk * LookaheadBatches, count) : chunk;
        return (chunk, reorder, window);
    }

    /// <summary>
    /// Stable ascending order of indices by width (Python's <c>list.sort</c> is stable, so ties keep input order).
    /// </summary>
    internal static int[] SortedOrder(IReadOnlyList<int> widths) =>
        [.. Enumerable.Range(0, widths.Count).OrderBy(i => widths[i])];

    /// <summary>
    /// Encode window by window, optionally sort each window by width, run chunk by chunk, and return
    /// the results in input order (onnx_agent.py:816-865).
    /// </summary>
    internal static IReadOnlyList<TResult> Execute<TEncoded, TResult>(
        IReadOnlyList<object?> states, int? batchSize, bool sortByLength,
        Func<IReadOnlyList<object?>, IReadOnlyList<TEncoded>> encode,
        Func<TEncoded, int> width,
        Func<IReadOnlyList<TEncoded>, IReadOnlyList<TResult>> infer)
    {
        if (states.Count == 0) return [];
        var (chunk, reorder, window) = Plan(states.Count, batchSize, sortByLength);
        var results = new List<TResult>(states.Count);
        // Sizes come from the input, never from the cap: Python accepts any positive batch_size.
        for (long start = 0; start < states.Count; start += window)
        {
            var end = (int)Math.Min(start + window, states.Count);
            var part = new List<object?>(end - (int)start);
            for (var i = (int)start; i < end; i++) part.Add(states[i]);
            var encoded = encode(part);
            var order = reorder
                ? SortedOrder([.. encoded.Select(width)])
                : Enumerable.Range(0, encoded.Count).ToArray();
            var windowResults = new TResult[encoded.Count];
            for (long offset = 0; offset < order.Length; offset += chunk)
            {
                var indices = order.Skip((int)offset).Take(chunk).ToArray();
                var chunkResults = infer([.. indices.Select(i => encoded[i])]);
                if (chunkResults.Count != indices.Length)
                    throw new InvalidOperationException(
                        $"internal error: a batch of {indices.Length} states produced {chunkResults.Count} results");
                for (var k = 0; k < indices.Length; k++) windowResults[indices[k]] = chunkResults[k];
            }
            results.AddRange(windowResults);
        }
        return results;
    }
}
