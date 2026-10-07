namespace Laya.Tests;

/// <summary>
/// Tier 1: sequence layout and the option budget, using <see cref="StubTokenizer"/> so no artifacts
/// are needed. Real token ids are checked in <see cref="TokenizerParityTests"/>.
/// </summary>
public sealed class SequenceBuilderTests
{
    private readonly StubTokenizer _tok = new();

    [Fact]
    public void LaysOutClsHeadSepOptionsSepStateSep()
    {
        var q = Question.Choice("pick one", ("a", null), ("b", null));
        var (ids, markers) = SequenceBuilder.Build(_tok, "state text", q, maxLen: 64, headMaxLen: 32);

        Assert.Equal(_tok.ClsId, ids[0]);
        Assert.Equal(_tok.SepId, ids[^1]);
        // Each marker is a mask token, and the two options are adjacent fragments.
        Assert.Equal(2, markers.Length);
        Assert.All(markers, m => Assert.Equal(_tok.MaskId, ids[m]));
        Assert.True(markers[1] > markers[0]);
        // A SEP closes the head, immediately before the first marker.
        Assert.Equal(_tok.SepId, ids[markers[0] - 1]);
    }

    [Fact]
    public void MarkerPositionsPointAtConsecutiveOptionFragments()
    {
        // One word per option keeps the arithmetic checkable: mask + 1 word = 2 ids per fragment.
        var q = Question.Choice("h", ("alpha", null), ("beta", null), ("gamma", null));
        var (ids, markers) = SequenceBuilder.Build(_tok, "s", q, maxLen: 64, headMaxLen: 32);

        Assert.Equal(3, markers.Length);
        Assert.Equal(markers[0] + 2, markers[1]);
        Assert.Equal(markers[1] + 2, markers[2]);
        Assert.Equal(_tok.SepId, ids[markers[2] + 2]);
    }

    [Fact]
    public void PutsTheTypeNameInTheHeadText()
    {
        SequenceBuilder.Build(_tok, "s", Question.Noul("does it hold"), maxLen: 64, headMaxLen: 32);
        Assert.Equal("noul question: does it hold", _tok.Encoded[0]);
    }

    [Theory]
    [InlineData(QuestionType.Choice, "choice")]
    [InlineData(QuestionType.Score, "score")]
    [InlineData(QuestionType.Noul, "noul")]
    public void TypeNamesMatchPython(QuestionType type, string expected) =>
        Assert.Equal(expected, SequenceBuilder.TypeName(type));

    [Fact]
    public void OptionFragmentsAreEncodedWithALeadingSpace()
    {
        SequenceBuilder.Build(_tok, "s", Question.Choice("h", ("a", null)), maxLen: 64, headMaxLen: 32);
        // Python encodes " " + option, which the Metaspace pre-tokenizer turns into a word boundary.
        Assert.Contains(" a", _tok.Encoded);
    }

    [Fact]
    public void RendersOptionsTheWayPythonDoes()
    {
        Assert.Equal(["a: first", "b"],
            SequenceBuilder.RenderOptions(Question.Choice("h", ("a", "first"), ("b", null))));

        Assert.Equal(["level 0: low", "level 1: high"],
            SequenceBuilder.RenderOptions(Question.Score("h", "low", "high")));

        Assert.Equal(["false: no, the statement does not hold", "true: yes, the statement holds"],
            SequenceBuilder.RenderOptions(Question.Noul("h")));

        Assert.Equal(["false: all fine", "true: on fire"],
            SequenceBuilder.RenderOptions(Question.Noul("h", ifFalse: "all fine", ifTrue: "on fire")));
    }

    [Fact]
    public void TreatsOnlyNullAndEmptyStringAsAMissingDescription()
    {
        // Python tests `v is None or v == ""`, so 0 and false are real descriptions.
        Assert.Equal(["zero: 0", "no: false", "blank", "none"],
            SequenceBuilder.RenderOptions(Question.Choice("h",
                ("zero", 0L), ("no", false), ("blank", ""), ("none", null))));
    }

    [Fact]
    public void ScrubsAMaskTokenOutOfEveryFragment()
    {
        // A literal <mask> in user input would forge a marker the scorer then reads as an option.
        var q = Question.Choice($"is this {_tok.MaskToken} safe", ($"a{_tok.MaskToken}b", null));
        SequenceBuilder.Build(_tok, $"state {_tok.MaskToken} here", q, maxLen: 64, headMaxLen: 32);

        Assert.All(_tok.Encoded, text => Assert.DoesNotContain(_tok.MaskToken, text));
    }

    [Fact]
    public void TruncatesLongOptionsToFortyEightTokens()
    {
        var words = string.Join(' ', Enumerable.Range(0, 120).Select(i => $"w{i}"));
        var q = Question.Choice("h", ("opt", words));
        var (ids, markers) = SequenceBuilder.Build(_tok, "s", q, maxLen: 512, headMaxLen: 256);

        // The only fragment is mask + at most 48 option tokens, then the closing SEP. "opt: " adds
        // two words, so 48 is the cap that bites rather than the option running out.
        Assert.Single(markers);
        var fragmentEnd = Array.IndexOf(ids, _tok.SepId, markers[0]);
        Assert.Equal(markers[0] + 49, fragmentEnd);
    }

    [Fact]
    public void ShrinksOptionsWhenTheyWouldEatTheWholeHeadBudget()
    {
        // 20 options of ~10 tokens each against a 96-token head budget forces the
        // `opt_budget < 16` branch, which caps every fragment at `(head_max_len - 16) // n`.
        var options = Enumerable.Range(0, 20)
            .Select(i => ($"opt{i}", (object?)string.Join(' ', Enumerable.Range(0, 10).Select(j => $"d{i}_{j}"))))
            .ToArray();
        var (ids, markers) = SequenceBuilder.Build(_tok, "s", Question.Choice("head words here", options),
                                                  maxLen: 512, headMaxLen: 96);

        Assert.Equal(20, markers.Length);
        var per = Math.Max(4, (96 - 16) / 20);   // 4
        for (var i = 1; i < markers.Length; i++)
            Assert.True(markers[i] - markers[i - 1] <= per,
                $"fragment {i - 1} is {markers[i] - markers[i - 1]} tokens, above the {per}-token cap");
        Assert.Equal(_tok.ClsId, ids[0]);
    }

    [Fact]
    public void KeepsAtLeastEightHeadTokensWhenOptionsDominate()
    {
        var options = Enumerable.Range(0, 30)
            .Select(i => ($"o{i}", (object?)"a b c d e f g h"))
            .ToArray();
        var (ids, markers) = SequenceBuilder.Build(_tok, "s",
            Question.Choice("one two three four five six seven eight nine ten", options),
            maxLen: 512, headMaxLen: 64);

        // CLS + head + SEP, so the first marker cannot be earlier than position 2.
        Assert.True(markers[0] >= 2);
        Assert.Equal(_tok.SepId, ids[markers[0] - 1]);
    }

    [Fact]
    public void TruncatesTheStateToFitMaxLen()
    {
        var state = string.Join(' ', Enumerable.Range(0, 500).Select(i => $"s{i}"));
        var (ids, markers) = SequenceBuilder.Build(_tok, state, Question.Noul("h"), maxLen: 64, headMaxLen: 32);

        Assert.Equal(64, ids.Length);
        Assert.Equal(_tok.SepId, ids[^1]);
        Assert.Equal(2, markers.Length);
    }

    [Fact]
    public void KeepsTheStateTailWhenTruncatingLeft()
    {
        var state = string.Join(' ', Enumerable.Range(0, 200).Select(i => $"s{i}"));
        var head = SequenceBuilder.Build(_tok, state, Question.Noul("h"), 64, 32, truncateLeft: false);
        var tail = SequenceBuilder.Build(_tok, state, Question.Noul("h"), 64, 32, truncateLeft: true);

        Assert.Equal(head.Ids.Length, tail.Ids.Length);
        Assert.NotEqual(head.Ids, tail.Ids);
        // Both keep the same prefix up to the state, and differ from there on.
        Assert.Equal(head.Markers, tail.Markers);
    }

    [Fact]
    public void LeftTruncationWithNoRoomDropsTheWholeStateRatherThanKeepingIt()
    {
        // 0.3.21's `state_ids[max(0, len - room):]` correctly returns empty when room == 0.
        // 0.3.6's `st[-room:]` had a quirk here (Python's `x[-0:]` is `x[0:]`, the whole list) —
        // reproduced faithfully until now, since `truncate_left` was previously unreachable from
        // Predict. Isolate the "no room left" boundary without hardcoding tokenizer counts: probe
        // the exact CLS+head+options+SEP prefix length with a one-token state, then rebuild at
        // exactly that maxLen (room == 0) with a long state.
        var q = Question.Noul("h");
        var probe = SequenceBuilder.Build(_tok, "x", q, maxLen: 10_000, headMaxLen: 32);
        var prefixLen = probe.Ids.Length - 2; // minus the one state token and the closing SEP

        var longState = string.Join(' ', Enumerable.Range(0, 50).Select(i => $"s{i}"));
        var (ids, _) = SequenceBuilder.Build(_tok, longState, q,
            maxLen: prefixLen + 1, headMaxLen: 32, truncateLeft: true);

        // Fixed: no state tokens fit, so the sequence ends prefix + SEP, exactly at maxLen.
        // The old bug would instead have appended the whole 50-word state before the final
        // maxLen clamp cut it back down to size, landing a state token (not SEP) at this position.
        Assert.Equal(prefixLen + 1, ids.Length);
        Assert.Equal(_tok.SepId, ids[^1]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ListStatesAreDetectedForAutomaticLeftTruncation(bool expectListLike)
    {
        // Mirrors Agent._encode_state: `truncate_left = isinstance(state, list)`. LayaEngine.Predict
        // derives its truncateLeft argument from PythonJson.IsListState, so this is the source of
        // truth callers rely on for "is this state list-shaped conversation turns".
        object state = expectListLike
            ? new List<object?> { "turn 1", "turn 2" }
            : "a plain string state";

        Assert.Equal(expectListLike, PythonJson.IsListState(state));
    }

    [Fact]
    public void DictAndStringStatesAreNotListLike()
    {
        Assert.False(PythonJson.IsListState("a string"));
        Assert.False(PythonJson.IsListState(new Dictionary<string, object?> { ["k"] = "v" }));
        Assert.False(PythonJson.IsListState(null));
        Assert.True(PythonJson.IsListState(new object?[] { "a", "b" }));
    }

    [Fact]
    public void DropsMarkersThatWouldLandPastMaxLen()
    {
        // 40 options cannot fit a 32-token sequence, so markers are lost. Predict must treat a short
        // marker list as an error rather than answering a question missing options.
        var options = Enumerable.Range(0, 40).Select(i => ($"o{i}", (object?)"desc")).ToArray();
        var q = Question.Choice("h", options);
        var (ids, markers) = SequenceBuilder.Build(_tok, "s", q, maxLen: 32, headMaxLen: 256);

        Assert.Equal(32, ids.Length);
        Assert.True(markers.Length < options.Length);
        Assert.All(markers, m => Assert.True(m < 32));
    }

    [Fact]
    public void RendersNonStringInstructionsWithRealUnicodeNotEscapes()
    {
        // 0.3.21's `_to_internal` switched to json.dumps(ins, ensure_ascii=False): real Unicode
        // glyphs reach the tokenizer instead of literal \uXXXX escape text.
        var q = Question.Noul(new OrderedDictionary<string, object?>(StringComparer.Ordinal)
        {
            ["hint"] = "Müller",
        });
        var rendered = SequenceBuilder.RenderInstructions(q);

        Assert.Contains("ü", rendered);
        Assert.DoesNotContain("\\u", rendered);
        Assert.Equal("{\"hint\": \"Müller\"}", rendered);
    }

    [Fact]
    public void PassesStringInstructionsThroughUntouched()
    {
        var q = Question.Noul("Müller & co <urgent>");
        Assert.Equal("Müller & co <urgent>", SequenceBuilder.RenderInstructions(q));
    }
}
