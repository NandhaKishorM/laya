namespace Laya;

/// <summary>
/// Turns raw marker logits into calibrated probabilities and confidence. Ports
/// <c>temp_bucket</c>, <c>clamp_temperature</c> and <c>confidence_from_probs</c> from
/// <c>laya/common.py</c>, plus the readout arithmetic in <c>Agent.system_one</c>.
/// </summary>
public static class Calibration
{
    /// <summary>
    /// Lowest temperature that will be applied. A fitted temperature below 1 sharpens logits
    /// instead of softening them; the shipped <c>choice:11+</c> bucket of 0.1006 multiplies them
    /// roughly tenfold, publishing a 0.24 top probability as 0.99. A caller gating on confidence
    /// would be told a coin flip is a certainty, so such a value is refused rather than applied.
    /// </summary>
    public const double TempMin = 0.5;

    /// <summary>Highest temperature that will be applied.</summary>
    public const double TempMax = 5.0;

    /// <summary>A usable temperature: confined to <see cref="TempMin"/>..<see cref="TempMax"/>, with
    /// anything that is not a finite number falling back to 1.0.</summary>
    public static double ClampTemperature(double t) =>
        double.IsNaN(t) || double.IsInfinity(t) ? 1.0 : Math.Min(TempMax, Math.Max(TempMin, t));

    /// <summary>
    /// The calibration bucket key for a question type and option count, e.g. <c>"choice:3-5"</c>.
    /// </summary>
    public static string TempBucket(QuestionType type, int optionCount)
    {
        var size = optionCount <= 2 ? "2"
            : optionCount <= 5 ? "3-5"
            : optionCount <= 10 ? "6-10"
            : "11+";
        return $"{SequenceBuilder.TypeName(type)}:{size}";
    }

    /// <summary>The temperature to divide logits by for one question.</summary>
    public static double ResolveTemperature(LayaConfig config, QuestionType type, int optionCount)
    {
        ArgumentNullException.ThrowIfNull(config);
        return config.TemperatureByOptions.TryGetValue(TempBucket(type, optionCount), out var t)
            ? t
            : config.Temperature[(int)type];
    }

    /// <summary>
    /// Tempered softmax over the first <paramref name="optionCount"/> logits of one row.
    /// </summary>
    /// <remarks>
    /// Only the real markers are read. Padding columns carry the <c>-1e4</c> the model's
    /// <c>masked_fill</c> wrote, which would underflow to zero anyway, but slicing first keeps the
    /// normalisation identical to Python's <c>logits[r, :k]</c>.
    /// </remarks>
    public static double[] Probabilities(ReadOnlySpan<float> logits, int optionCount, double temperature)
    {
        if (optionCount <= 0) throw new ArgumentOutOfRangeException(nameof(optionCount));

        var p = new double[optionCount];
        var max = double.NegativeInfinity;
        for (var i = 0; i < optionCount; i++)
        {
            p[i] = logits[i] / temperature;
            if (p[i] > max) max = p[i];
        }

        var sum = 0.0;
        for (var i = 0; i < optionCount; i++)
        {
            p[i] = Math.Exp(p[i] - max);
            sum += p[i];
        }
        for (var i = 0; i < optionCount; i++) p[i] /= sum;
        return p;
    }

    /// <summary>Softmax over a whole row, used for the action head.</summary>
    public static double[] Softmax(ReadOnlySpan<float> logits)
    {
        var p = new double[logits.Length];
        var max = double.NegativeInfinity;
        foreach (var z in logits) if (z > max) max = z;

        var sum = 0.0;
        for (var i = 0; i < logits.Length; i++)
        {
            p[i] = Math.Exp(logits[i] - max);
            sum += p[i];
        }
        for (var i = 0; i < p.Length; i++) p[i] /= sum;
        return p;
    }

    /// <summary>
    /// Normalized Shannon entropy confidence, <c>1 - H(p) / log k</c>, clipped to [0, 1].
    /// A single-option question is fully decided by construction, so it returns 1.0.
    /// </summary>
    public static double ConfidenceFromProbs(ReadOnlySpan<double> p, int optionCount)
    {
        if (optionCount < 2) return 1.0;

        var entropy = 0.0;
        for (var i = 0; i < optionCount; i++)
        {
            var clipped = Math.Clamp(p[i], 1e-12, 1.0);
            entropy -= p[i] * Math.Log(clipped);
        }
        return Math.Clamp(1.0 - entropy / Math.Log(optionCount), 0.0, 1.0);
    }

    /// <summary>
    /// The calibrated <c>answer_confidence</c>: <c>max(p[:k])</c>, clipped to [0, 1]. A
    /// single-option question is fully decided by construction, so it returns 1.0.
    /// </summary>
    /// <remarks>
    /// Ports <c>answer_confidence</c> in <c>laya/common.py</c> (0.3.21, surfaced via the new
    /// <c>laya/confidence.py</c>). Unlike <see cref="ConfidenceFromProbs"/> — normalized Shannon
    /// entropy for choice/score, <c>max(p, 1-p)</c> for noul — this is the one quantity
    /// temperature scaling fits and every calibration figure is computed on, so it means the same
    /// thing across all three question types and is safe to gate a single threshold on. Since
    /// <paramref name="p"/> here already holds exactly <paramref name="optionCount"/> entries
    /// (see <see cref="Probabilities"/>), this is just <c>max(p)</c>.
    /// </remarks>
    public static double AnswerConfidence(ReadOnlySpan<double> p, int optionCount)
    {
        if (optionCount < 1) return 1.0;

        var max = double.NegativeInfinity;
        for (var i = 0; i < optionCount; i++) if (p[i] > max) max = p[i];
        return Math.Clamp(max, 0.0, 1.0);
    }

    /// <summary>
    /// The expected level of a score answer: the probability-weighted mean of the level indices,
    /// so a fractional value rather than the most likely level.
    /// </summary>
    public static double ExpectedScore(ReadOnlySpan<double> p)
    {
        var expected = 0.0;
        for (var i = 0; i < p.Length; i++) expected += i * p[i];
        return expected;
    }

    /// <summary>The index of the largest probability, ties going to the lowest index as NumPy's
    /// <c>argmax</c> does.</summary>
    public static int ArgMax(ReadOnlySpan<double> p)
    {
        var best = 0;
        for (var i = 1; i < p.Length; i++) if (p[i] > p[best]) best = i;
        return best;
    }

    /// <summary>
    /// Python's <c>round(x, 4)</c>. Both runtimes round half to even on the binary value, so this
    /// agrees with the reference implementation.
    /// </summary>
    public static double Round4(double value) => Math.Round(value, 4, MidpointRounding.ToEven);
}
