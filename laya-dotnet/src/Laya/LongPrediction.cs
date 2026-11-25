using System.Diagnostics;
using Laya.Tokenization;

namespace Laya;

internal static class LongPrediction
{
    internal static (int Window, int Stride, int Room) Budget(
        ILayaTokenizer tokenizer, QuestionSet questions, int maxLen, int headMaxLen,
        int? window, int? stride)
    {
        var requested = window is > 0 ? window.Value : Math.Max(64, maxLen - headMaxLen - 8);
        var room = questions.Count == 0 ? requested : questions.Ids.Min(
            id => SequenceBuilder.StateRoom(tokenizer, questions[id], maxLen, headMaxLen));
        return BudgetForRoom(requested, room, maxLen, headMaxLen, window, stride);
    }

    // window_budget once the smallest room is known (common.py:404 onward).
    internal static (int Window, int Stride, int Room) BudgetForRoom(
        int requested, int room, int maxLen, int headMaxLen, int? window, int? stride)
    {
        if (room <= 0)
            throw new ArgumentException(
                $"predict_long: the questions' options fill the whole sequence (max_len={maxLen},"
                + $" head_max_len={headMaxLen}), leaving no room for the state; no window can carry any of it."
                + " A label set this large is what laya.shortlist.predict_shortlist is for", "questions");
        var size = Math.Min(requested, room);
        // common.py:410-440: the two texts Python warns with when the window is cut to the room.
        if (size < requested && window is > 0)
            WarnOnce.Warn(
                $"laya: predict_long: window={requested} is wider than the {room} state tokens these questions"
                + $" leave inside max_len={maxLen}, so every window would be truncated to {room} on the way to"
                + $" the model; scanning with window={room} instead");
        else if (size < requested && requested >= room * 2L)
            WarnOnce.Warn(
                $"laya: predict_long: these questions leave only {room} of max_len={maxLen} for the state"
                + $" (their heads take the rest), so the scan window is capped {requested} -> {room} and roughly"
                + $" {PythonFormat1((double)requested / room)}x as many windows -- each a full forward pass -- are needed to read the"
                + " document. Fewer or shorter options, a larger max_len, or"
                + " laya.shortlist.predict_shortlist for a large label set will all cost less than"
                + " scanning at this width");
        var step = stride is > 0 ? stride.Value : Math.Max(1, size / 2);
        if (step > size)
        {
            if (size < requested && stride <= requested)
            {
                var reduced = Math.Max(1, size / 2);
                WarnOnce.Warn(
                    $"laya: predict_long: stride={step} was a 50% step for the window={requested} you asked for, but"
                    + $" the window was reduced to {size} to fit the room these questions leave; scanning with"
                    + $" stride={reduced} instead");
                step = reduced;
            }
            else
                throw new ArgumentException(
                    $"predict_long: stride={step} steps past the {size}-token window"
                    + (size < requested ? " these questions leave room for" : "")
                    + $", so {step - size} tokens between every pair of windows would be read by no window at all;"
                    + $" pass stride <= {size}", nameof(stride));
        }
        return (size, step, room);
    }

    // Python's "%.1f": the exact binary value rounded to one decimal, a tie going to the even digit
    // (.NET's "F1" breaks ties away from zero, which differs for x.25 and x.75).
    internal static string PythonFormat1(double value)
    {
        var scaled = value * 10;
        var floor = Math.Floor(scaled);
        // an exact tie needs a multiple of 0.25, whose product with 10 is exact
        if (scaled - floor == 0.5 && value * 4 == Math.Floor(value * 4))
            return ((floor % 2 == 0 ? floor : floor + 1) / 10).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
        return value.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
    }

    internal static IReadOnlyList<WindowInfo> Spans(int tokens, int window, int stride)
    {
        var starts = new List<int>();
        for (var i = 0; i < tokens;)
        {
            starts.Add(i);
            if ((long)i + window >= tokens) break;
            i += stride;
        }
        return starts.Select((start, index) =>
            new WindowInfo(index, start, (int)Math.Min((long)start + window, tokens), starts.Count)).ToArray();
    }

    internal static int? BatchCap(int count, int window, int configBudget, int? batchSize)
    {
        if (batchSize is > 0) return batchSize;
        if (window >= configBudget) return null;
        var step = Math.Max(1, window / 2);
        var configStep = Math.Max(1, configBudget / 2);
        var unclamped = Math.Max(1, (int)(((long)count * step + configStep - 1) / configStep));
        return count <= unclamped * 2L ? null : unclamped;
    }

    internal static LayaResult Aggregate(
        QuestionSet questions, IReadOnlyList<LayaResult> results, IReadOnlyList<WindowInfo> spans)
    {
        var answers = new Dictionary<string, Answer>(StringComparer.Ordinal);
        foreach (var id in questions.Ids)
        {
            var best = 0;
            double Evidence(int index) => questions[id].Type == QuestionType.Noul
                ? results[index][id].AsNoul().Probability : results[index][id].AnswerConfidence;
            for (var i = 1; i < results.Count; i++)
                if (Evidence(i) > Evidence(best)) best = i;
            answers[id] = results[best][id].WithWindow(spans[best]);
        }
        Dictionary<string, OptionUsage>? options = null;
        foreach (var result in results)
            if (result.Usage.Options is { } records)
            {
                options ??= new(StringComparer.Ordinal);
                foreach (var (id, stats) in records) options[id] = stats;
            }
        var usage = new Usage(results.Sum(r => r.Usage.InputTokens), 0)
        {
            Windows = results.Count,
            StateTokens = results.Any(r => r.Usage.StateTokens.HasValue)
                ? results.Sum(r => r.Usage.StateTokens ?? 0) : null,
            StateTokensDropped = results.Any(r => r.Usage.StateTokensDropped.HasValue)
                ? results.Sum(r => r.Usage.StateTokensDropped ?? 0) : null,
            Truncated = results.Any(r => r.Usage.Truncated.HasValue)
                ? results.Sum(r => r.Usage.Truncated ?? 0) : null,
            TruncatedQuestions = results.Count == 0 ? null : results[^1].Usage.TruncatedQuestions,
            Options = options,
        };
        return new LayaResult("laya-rl-agent", [.. questions.Ids], answers, usage);
    }
}
