using Laya.Tokenization;

namespace Laya;

/// <summary>
/// Builds the typed token sequence one question is answered from. Ports <c>build_sequence</c>,
/// <c>render_options</c> and <c>render_criterion</c> from <c>laya/common.py</c>.
/// </summary>
/// <remarks>
/// The layout is
/// <c>[CLS] &lt;type&gt; question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]</c>.
/// The scorer reads the hidden state at each <c>[MASK]</c>, so a marker position that is off by one
/// token attaches an option's probability to the wrong place in the sequence and the model answers
/// a different question without anything failing. Every offset here is therefore load-bearing.
/// </remarks>
public static class SequenceBuilder
{
    /// <summary>One option fragment is capped at this many tokens, before the marker is prepended.</summary>
    private const int MaxOptionTokens = 48;

    /// <summary>The head is never squeezed below this, even when the options have eaten the budget.</summary>
    private const int MinHeadTokens = 8;

    /// <summary>The Python type name, which appears verbatim in the prompt text.</summary>
    public static string TypeName(QuestionType type) => type switch
    {
        QuestionType.Choice => "choice",
        QuestionType.Score => "score",
        QuestionType.Noul => "noul",
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "unknown question type"),
    };

    /// <summary>
    /// The instruction text as it reaches the tokenizer: a string verbatim, anything else through
    /// the <c>json.dumps</c> dialect that escapes non-ASCII. Ports <c>Agent._to_internal</c>.
    /// </summary>
    public static string RenderInstructions(Question question)
    {
        ArgumentNullException.ThrowIfNull(question);
        return question.Instructions as string ?? PythonJson.Instructions(question.Instructions);
    }

    /// <summary>
    /// The option texts in label-index order. Index N is the option the Nth logit scores, so this
    /// order is what binds a probability to a label.
    /// </summary>
    public static IReadOnlyList<string> RenderOptions(Question question)
    {
        ArgumentNullException.ThrowIfNull(question);
        switch (question)
        {
            case ChoiceQuestion choice:
                var rendered = new List<string>(choice.Options.Count);
                foreach (var (label, description) in choice.Options)
                    rendered.Add(IsBlankCriterion(description)
                        ? label
                        : $"{label}: {PythonJson.Criterion(description)}");
                return rendered;

            case ScoreQuestion score:
                var levels = new List<string>(score.Levels.Count);
                for (var i = 0; i < score.Levels.Count; i++)
                    levels.Add($"level {i}: {PythonJson.Criterion(score.Levels[i])}");
                return levels;

            case NoulQuestion noul:
                return
                [
                    "false: " + (IsBlankCriterion(noul.IfFalse)
                        ? "no, the statement does not hold"
                        : PythonJson.Criterion(noul.IfFalse)),
                    "true: " + (IsBlankCriterion(noul.IfTrue)
                        ? "yes, the statement holds"
                        : PythonJson.Criterion(noul.IfTrue)),
                ];

            default:
                throw new ArgumentException($"unknown question kind {question.GetType().Name}", nameof(question));
        }
    }

    /// <summary>
    /// Whether a criterion value means "no description". Python tests <c>v is None or v == ""</c>,
    /// so only null and the empty string qualify: <c>0</c> and <c>false</c> are real descriptions
    /// and get rendered.
    /// </summary>
    private static bool IsBlankCriterion(object? value) => value is null || (value is string s && s.Length == 0);

    /// <summary>Build the token sequence and marker positions for one question.</summary>
    /// <param name="tokenizer">Must encode without special tokens.</param>
    /// <param name="state">The text, object or turn list being judged.</param>
    /// <param name="question">The question to build for.</param>
    /// <param name="maxLen">Total sequence budget.</param>
    /// <param name="headMaxLen">Budget for instructions plus option fragments.</param>
    /// <param name="truncateLeft">Keep the end of an over-long state rather than its start.</param>
    /// <returns>
    /// The token ids, and one marker position per option. Markers landing at or past
    /// <paramref name="maxLen"/> are dropped, so a short marker list means options did not fit —
    /// the caller must treat that as an error rather than answering a truncated question.
    /// </returns>
    public static (int[] Ids, int[] Markers) Build(
        ILayaTokenizer tokenizer, object? state, Question question,
        int maxLen, int headMaxLen, bool truncateLeft = false)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(question);

        var mask = tokenizer.MaskToken;
        var options = RenderOptions(question);

        // A literal mask token anywhere in user input would forge a marker the scorer then reads,
        // so every fragment has it replaced with a space before encoding.
        var instructions = Scrub(RenderInstructions(question), mask);
        var headIds = tokenizer.Encode($"{TypeName(question.Type)} question: {instructions}").ToList();

        var optionIds = new List<List<int>>(options.Count);
        foreach (var option in options)
        {
            var encoded = tokenizer.Encode(" " + Scrub(option, mask));
            var take = Math.Min(MaxOptionTokens, encoded.Length);
            var fragment = new List<int>(take + 1) { tokenizer.MaskId };
            for (var i = 0; i < take; i++) fragment.Add(encoded[i]);
            optionIds.Add(fragment);
        }

        var optBudget = headMaxLen - optionIds.Sum(o => o.Count);
        if (optBudget < 16)
        {
            var per = Math.Max(4, FloorDiv(headMaxLen - 16, Math.Max(1, optionIds.Count)));
            for (var i = 0; i < optionIds.Count; i++)
                if (optionIds[i].Count > per)
                    optionIds[i] = optionIds[i][..per];
            optBudget = headMaxLen - optionIds.Sum(o => o.Count);
        }

        var headTake = Math.Max(MinHeadTokens, optBudget);
        if (headIds.Count > headTake) headIds = headIds[..headTake];

        var ids = new List<int>(maxLen) { tokenizer.ClsId };
        ids.AddRange(headIds);
        ids.Add(tokenizer.SepId);

        var markers = new List<int>(optionIds.Count);
        foreach (var fragment in optionIds)
        {
            markers.Add(ids.Count);
            ids.AddRange(fragment);
        }
        ids.Add(tokenizer.SepId);

        var room = Math.Max(0, maxLen - ids.Count - 1);
        var stateIds = tokenizer.Encode(Scrub(PythonJson.State(state), mask));
        ids.AddRange(SliceState(stateIds, room, truncateLeft));
        ids.Add(tokenizer.SepId);

        if (ids.Count > maxLen) ids = ids[..maxLen];
        markers.RemoveAll(m => m >= maxLen);
        return (ids.ToArray(), markers.ToArray());
    }

    private static string Scrub(string text, string maskToken) =>
        text.Replace(maskToken, " ", StringComparison.Ordinal);

    private static IEnumerable<int> SliceState(int[] stateIds, int room, bool truncateLeft)
    {
        if (!truncateLeft) return stateIds.Take(room);
        // Matches 0.3.21's `state_ids[max(0, len(state_ids) - room):]`. 0.3.6 wrote `st[-room:]`,
        // which for room == 0 is Python's `st[0:]` and so kept the whole state instead of none of
        // it; `Skip(Math.Max(0, len - room))` has no such quirk and correctly returns empty when
        // room == 0. `Agent._encode_state` sets `truncate_left = isinstance(state, list)`, so this
        // path is reachable whenever a caller's state is a list of conversation turns.
        return stateIds.Skip(Math.Max(0, stateIds.Length - room));
    }

    /// <summary>Python's <c>//</c>, which floors rather than truncating toward zero.</summary>
    private static int FloorDiv(int a, int b)
    {
        var q = a / b;
        return a % b != 0 && (a < 0) != (b < 0) ? q - 1 : q;
    }
}
