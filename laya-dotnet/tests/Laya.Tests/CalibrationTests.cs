namespace Laya.Tests;

/// <summary>Tier 1: the calibration maths, which needs no artifacts.</summary>
public sealed class CalibrationTests
{
    [Theory]
    [InlineData(QuestionType.Choice, 1, "choice:2")]
    [InlineData(QuestionType.Choice, 2, "choice:2")]
    [InlineData(QuestionType.Choice, 3, "choice:3-5")]
    [InlineData(QuestionType.Choice, 5, "choice:3-5")]
    [InlineData(QuestionType.Choice, 6, "choice:6-10")]
    [InlineData(QuestionType.Choice, 10, "choice:6-10")]
    [InlineData(QuestionType.Choice, 11, "choice:11+")]
    [InlineData(QuestionType.Score, 4, "score:3-5")]
    [InlineData(QuestionType.Noul, 2, "noul:2")]
    public void TempBucketMatchesPythonBoundaries(QuestionType type, int k, string expected) =>
        Assert.Equal(expected, Calibration.TempBucket(type, k));

    [Theory]
    [InlineData(1.0, 1.0)]
    [InlineData(2.5, 2.5)]
    [InlineData(0.1006, Calibration.TempMin)]   // the shipped choice:11+ value, refused as too sharp
    [InlineData(0.0, Calibration.TempMin)]
    [InlineData(-3.0, Calibration.TempMin)]
    [InlineData(99.0, Calibration.TempMax)]
    [InlineData(double.NaN, 1.0)]
    [InlineData(double.PositiveInfinity, 1.0)]
    [InlineData(double.NegativeInfinity, 1.0)]
    public void ClampTemperatureMatchesPython(double input, double expected) =>
        Assert.Equal(expected, Calibration.ClampTemperature(input));

    [Fact]
    public void ProbabilitiesReadOnlyTheRealMarkers()
    {
        // The padded marker column carries the -1e4 the model's masked_fill wrote. A single-option
        // question must still come out as a certainty rather than leaking the pad column in.
        double[] p = Calibration.Probabilities([-0.9127780f, -10000.0f], optionCount: 1, temperature: 1.0);
        Assert.Single(p);
        Assert.Equal(1.0, p[0], 12);
    }

    [Fact]
    public void ProbabilitiesSumToOneAndRespectTemperature()
    {
        float[] logits = [2.0f, 1.0f, 0.0f];
        var sharp = Calibration.Probabilities(logits, 3, 0.5);
        var soft = Calibration.Probabilities(logits, 3, 5.0);

        Assert.Equal(1.0, sharp.Sum(), 12);
        Assert.Equal(1.0, soft.Sum(), 12);
        // A lower temperature concentrates mass on the top option; a higher one flattens it.
        Assert.True(sharp[0] > soft[0]);
        Assert.True(soft[2] > sharp[2]);
    }

    [Fact]
    public void ProbabilitiesSurviveLargeLogits()
    {
        // act_logits in the goldens reach ±1600; a naive exp() would overflow to NaN.
        var p = Calibration.Softmax([1366.2135f, -1624.665f]);
        Assert.Equal(1.0, p[0], 12);
        Assert.Equal(0.0, p[1], 12);
    }

    [Fact]
    public void ConfidenceIsOneForASingleOption() =>
        Assert.Equal(1.0, Calibration.ConfidenceFromProbs([1.0], optionCount: 1));

    [Fact]
    public void ConfidenceIsZeroForAUniformDistribution()
    {
        Assert.Equal(0.0, Calibration.ConfidenceFromProbs([0.5, 0.5], 2), 12);
        Assert.Equal(0.0, Calibration.ConfidenceFromProbs([0.25, 0.25, 0.25, 0.25], 4), 12);
    }

    [Fact]
    public void ConfidenceIsOneForACertainty() =>
        Assert.Equal(1.0, Calibration.ConfidenceFromProbs([1.0, 0.0], 2), 12);

    [Fact]
    public void ConfidenceMatchesTheRecordedQuickstartValues()
    {
        // From golden/multilingual/case_quickstart.json: urgency's four-level distribution and its confidence.
        double[] urgency = [0.0179, 0.465, 0.3208, 0.1964];
        Assert.Equal(0.1976, Calibration.Round4(Calibration.ConfidenceFromProbs(urgency, 4)), 3);
    }

    [Fact]
    public void ConfidenceIgnoresPaddingBeyondTheOptionCount()
    {
        // Passing a longer buffer must not change the answer: only the first k entries count.
        double[] padded = [0.5, 0.5, 0.0, 0.0];
        Assert.Equal(Calibration.ConfidenceFromProbs([0.5, 0.5], 2),
                     Calibration.ConfidenceFromProbs(padded, 2));
    }

    [Fact]
    public void AnswerConfidenceIsOneWhenThereIsOnlyOneOption() =>
        Assert.Equal(1.0, Calibration.AnswerConfidence([1.0], optionCount: 1));

    [Fact]
    public void AnswerConfidenceIsTheMaxProbability()
    {
        // Ports common.answer_confidence: float(np.clip(np.max(p[:k]), 0.0, 1.0)).
        double[] urgency = [0.0179, 0.465, 0.3208, 0.1964];
        Assert.Equal(0.465, Calibration.AnswerConfidence(urgency, 4), 12);
    }

    [Fact]
    public void AnswerConfidenceIgnoresPaddingBeyondTheOptionCount()
    {
        double[] padded = [0.2, 0.3, 0.9, 0.9];
        Assert.Equal(0.3, Calibration.AnswerConfidence(padded, 2), 12);
    }

    [Fact]
    public void AnswerConfidenceDiffersFromConfidenceFromProbsOnTheSameDistribution()
    {
        // Two genuinely distinct fields on a skewed distribution: ConfidenceFromProbs is a
        // normalized-entropy measure, AnswerConfidence is a bare max(p) - they must not collapse
        // to the same number here, or Answers.AnswerConfidence would be a redundant duplicate.
        double[] urgency = [0.0179, 0.465, 0.3208, 0.1964];
        var entropyConfidence = Calibration.ConfidenceFromProbs(urgency, 4);
        var answerConfidence = Calibration.AnswerConfidence(urgency, 4);

        Assert.Equal(0.1976, Calibration.Round4(entropyConfidence), 3);
        Assert.Equal(0.465, answerConfidence, 12);
        Assert.NotEqual(entropyConfidence, answerConfidence, 3);
    }

    [Fact]
    public void ExpectedScoreIsAProbabilityWeightedMean()
    {
        double[] urgency = [0.0179, 0.465, 0.3208, 0.1964];
        Assert.Equal(1.6956, Calibration.Round4(Calibration.ExpectedScore(urgency)), 3);
        Assert.Equal(0.0, Calibration.ExpectedScore([1.0, 0.0, 0.0]), 12);
        Assert.Equal(2.0, Calibration.ExpectedScore([0.0, 0.0, 1.0]), 12);
    }

    [Fact]
    public void ArgMaxBreaksTiesTowardTheLowestIndexAsNumpyDoes() =>
        Assert.Equal(0, Calibration.ArgMax([0.5, 0.5, 0.0]));

    [Fact]
    public void ResolveTemperaturePrefersTheBucketThenTheTypeDefault()
    {
        var config = LayaConfig.Parse("""
            {"max_len": 1024, "head_max_len": 256,
             "temperature": [1.5, 2.0, 2.5],
             "temperature_by_options": {"choice:3-5": 3.0, "noul:2": 0.2}}
            """);

        Assert.Equal(3.0, Calibration.ResolveTemperature(config, QuestionType.Choice, 4));
        Assert.Equal(1.5, Calibration.ResolveTemperature(config, QuestionType.Choice, 8));
        Assert.Equal(2.0, Calibration.ResolveTemperature(config, QuestionType.Score, 4));
        // 0.2 is below TempMin, so the bucket is clamped rather than applied as recorded.
        Assert.Equal(Calibration.TempMin, Calibration.ResolveTemperature(config, QuestionType.Noul, 2));
    }

    [Fact]
    public void ConfigFallsBackToPythonsInCodeDefaults()
    {
        var config = LayaConfig.Parse("{}");
        Assert.Equal(512, config.MaxLen);
        Assert.Equal(192, config.HeadMaxLen);
        Assert.Equal([1.0, 1.0, 1.0], config.Temperature);
    }

    [Fact]
    public void ShippedConfigParsesAsRecorded()
    {
        // meta.json records the config the goldens were produced under.
        var meta = GoldenData.Meta;
        var config = LayaConfig.Parse(meta.GetProperty("config").GetRawText());

        Assert.Equal(meta.GetProperty("max_len").GetInt32(), config.MaxLen);
        Assert.Equal(meta.GetProperty("head_max_len").GetInt32(), config.HeadMaxLen);
        Assert.Equal([.. meta.GetProperty("temperature").EnumerateArray().Select(e => e.GetDouble())],
                     config.Temperature);
        Assert.Empty(config.TemperatureByOptions);
    }
}
