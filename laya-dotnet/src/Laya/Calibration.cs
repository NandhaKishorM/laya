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
    /// <see cref="ClampTemperature(double)"/> for a value read from JSON, exactly as Python's
    /// <c>clamp_temperature</c> treats it: a <see cref="bool"/> is 1.0, and so is anything
    /// <c>float()</c> refuses (<see langword="null"/>, a list, a dictionary, a non-numeric string).
    /// An integer too wide for a double (Python's <c>OverflowError</c>) is 1.0 as well.
    /// A numeric string such as <c>"2.5"</c>, <c>" 2.5 "</c> or <c>"1e1"</c> is read the way
    /// <c>float()</c> reads it.
    /// </summary>
    public static double ClampTemperature(object? t)
    {
        if (t is bool) return 1.0;
        return PythonFloat.TryConvert(t, out var value) ? ClampTemperature(value) : 1.0;
    }

    /// <summary>
    /// Python's <c>temperature_rejected</c>: whether <paramref name="raw"/> is not the number
    /// <see cref="ClampTemperature(object?)"/> turned into <paramref name="applied"/>. A bool, or a
    /// value that will not convert, is rejected; so is a number that had to move.
    /// </summary>
    public static bool TemperatureRejected(object? raw, double applied)
    {
        if (raw is bool) return true;
        return !PythonFloat.TryConvert(raw, out var value) || value != applied;
    }

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
        return ResolveTemperature(config.Temperature, config.TemperatureByOptions, type, optionCount);
    }

    /// <summary><see cref="ResolveTemperature(LayaConfig, QuestionType, int)"/> over temperatures an engine holds, a calibration file's among them.</summary>
    internal static double ResolveTemperature(
        IReadOnlyList<double> temperatures, IReadOnlyDictionary<string, double> byOptions, QuestionType type, int optionCount)
    {
        // `self.temperature_by_options.get(bucket, self.temperature[qt])` evaluates the default first, so a
        // list too short for the question's type fails even when the bucket is present (onnx_agent.py:727).
        if ((int)type >= temperatures.Count)
            throw new InvalidOperationException("IndexError: list index out of range");
        return byOptions.TryGetValue(TempBucket(type, optionCount), out var t)
            ? t
            : temperatures[(int)type];
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
    /// Python's <c>round(x, 4)</c> on a float: the exact binary value is rounded to four decimals,
    /// half to even on exact ties, and the nearest double to that decimal is returned. Scaling by
    /// 10^4 and rounding in floating point does not do this (the product is itself rounded), so the
    /// rounding is done on the integer mantissa. NaN, infinities and a zero result's sign are kept.
    /// </summary>
    public static double Round4(double value)
    {
        var bits = BitConverter.DoubleToUInt64Bits(value);
        var biased = (int)((bits >> 52) & 0x7FF);
        if (biased == 0x7FF) return value; // NaN, +/-infinity
        // From 2^39 up a double is coarser than 1e-4, so the rounded decimal maps back to the value.
        if (biased - 1023 >= 39) return value;

        var mantissa = bits & 0x000F_FFFF_FFFF_FFFFUL;
        int exponent; // value = mantissa * 2^exponent
        if (biased == 0) exponent = -1074;
        else
        {
            mantissa |= 1UL << 52;
            exponent = biased - 1075;
        }

        var negative = (bits >> 63) != 0;
        var shift = -exponent; // exponent < 0 here: biased - 1075 < 39 - 1023 + 1023 - 52
        UInt128 quotient;
        if (shift >= 128) quotient = 0; // |x| * 10^4 < 2^67 / 2^128, far below one half
        else
        {
            var scaled = (UInt128)mantissa * 10_000; // < 2^67, exact
            quotient = scaled >> shift;
            var remainder = scaled - (quotient << shift);
            var half = (UInt128)1 << (shift - 1);
            if (remainder > half || (remainder == half && (quotient & 1) != 0)) quotient++;
        }

        // quotient < 2^39 * 10^4 < 2^53, so it converts exactly; one IEEE division then yields the
        // double nearest the decimal, as Python's string round trip does.
        var result = (double)(ulong)quotient / 10_000.0;
        return negative ? -result : result;
    }
}
