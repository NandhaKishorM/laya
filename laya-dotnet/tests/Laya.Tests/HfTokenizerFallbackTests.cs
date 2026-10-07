using Laya.Tokenization;

namespace Laya.Tests;

/// <summary>
/// Tier 1: <see cref="HfTokenizer"/>'s special-token resolution against a minimal, synthetic
/// <c>tokenizer.json</c> in a temp directory, covering the split layout's missing
/// <c>tokenizer_config.json</c> without needing a real checkpoint.
/// </summary>
/// <remarks>
/// The fixture below is not a usable tokenizer (its "vocabulary" only has enough entries to prove
/// which ids got resolved), but it is enough to exercise <c>HfTokenizer</c>'s constructor end to
/// end: the native <c>Tokenizers.DotNet</c> wrapper does load it, since a WordPiece model only
/// needs a <c>vocab</c> map and an <c>unk_token</c> to be valid.
/// </remarks>
public sealed class HfTokenizerFallbackTests : IDisposable
{
    private readonly string _dir;

    public HfTokenizerFallbackTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "laya-tokenizer-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); }
        catch (IOException) { /* best effort */ }
    }

    private static string AddedToken(string content, int id) =>
        $$"""{"id": {{id}}, "content": "{{content}}", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}""";

    /// <summary>A minimal but loadable WordPiece tokenizer.json using ModernBERT-style bracketed tokens.</summary>
    private const string BracketedVocab =
        """{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"hello":5}""";

    private void WriteTokenizerJson(string addedTokensJson, string vocabJson)
    {
        var json = $$"""
            {
              "version": "1.0",
              "truncation": null,
              "padding": null,
              "added_tokens": [{{addedTokensJson}}],
              "normalizer": null,
              "pre_tokenizer": {"type": "Whitespace"},
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {{vocabJson}}
              }
            }
            """;
        File.WriteAllText(Path.Combine(_dir, "tokenizer.json"), json);
    }

    [Fact]
    public void NoTokenizerConfig_ResolvesBracketedSpecialTokensFromAddedTokens()
    {
        // Split layout: no tokenizer_config.json beside tokenizer.json. Every role must be
        // resolved purely from added_tokens' conventional spellings.
        var added = string.Join(",", [
            AddedToken("[PAD]", 0), AddedToken("[UNK]", 1), AddedToken("[CLS]", 2),
            AddedToken("[SEP]", 3), AddedToken("[MASK]", 4),
        ]);
        WriteTokenizerJson(added, BracketedVocab);

        using var tok = new HfTokenizer(Path.Combine(_dir, "tokenizer.json"));

        Assert.Equal(0, tok.PadId);
        Assert.Equal(1, tok.UnkId);
        Assert.Equal(2, tok.ClsId);
        Assert.Equal(3, tok.SepId);
        Assert.Equal(4, tok.MaskId);
        Assert.Equal("[MASK]", tok.MaskToken);
    }

    [Fact]
    public void NoTokenizerConfig_ResolvesBosEosSpellingForClsAndSep()
    {
        // Some split exports use the BOS/EOS convention for CLS/SEP instead of the bracketed
        // ModernBERT form (e.g. mmBERT-style vocabularies). FirstKnownToken must fall through to
        // that spelling when the bracketed one is absent.
        var vocab = """{"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3,"<mask>":4,"hello":5}""";
        var added = string.Join(",", [
            AddedToken("<pad>", 0), AddedToken("<unk>", 1), AddedToken("<bos>", 2),
            AddedToken("<eos>", 3), AddedToken("<mask>", 4),
        ]);
        WriteTokenizerJson(added, vocab);

        using var tok = new HfTokenizer(Path.Combine(_dir, "tokenizer.json"));

        Assert.Equal(0, tok.PadId);
        Assert.Equal(1, tok.UnkId);
        Assert.Equal(2, tok.ClsId);
        Assert.Equal(3, tok.SepId);
        Assert.Equal(4, tok.MaskId);
        Assert.Equal("<mask>", tok.MaskToken);
    }

    [Fact]
    public void NoTokenizerConfig_TakesClsAndSepFromThePostProcessorTemplate()
    {
        // The real multilingual (mmBERT) vocabulary holds "<s>"/"</s>" as ordinary tokens as well
        // as its real CLS/SEP "<bos>"/"<eos>" (ids 2 and 1). The post_processor template names the
        // right pair, so it must win over any spelling guess.
        var vocab = """{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"<mask>":4,"<s>":5,"</s>":6,"hello":7}""";
        var added = string.Join(",", [
            AddedToken("<pad>", 0), AddedToken("<eos>", 1), AddedToken("<bos>", 2),
            AddedToken("<unk>", 3), AddedToken("<mask>", 4), AddedToken("<s>", 5), AddedToken("</s>", 6),
        ]);
        WriteTokenizerJson(added, vocab);
        var path = Path.Combine(_dir, "tokenizer.json");
        File.WriteAllText(path, File.ReadAllText(path).Replace("\"post_processor\": null", """
            "post_processor": {"type": "TemplateProcessing", "single": [
              {"SpecialToken": {"id": "<bos>", "type_id": 0}},
              {"Sequence": {"id": "A", "type_id": 0}},
              {"SpecialToken": {"id": "<eos>", "type_id": 0}}]}
            """));

        using var tok = new HfTokenizer(path);

        Assert.Equal(2, tok.ClsId);
        Assert.Equal(1, tok.SepId);
        Assert.Equal(0, tok.PadId);
        Assert.Equal(3, tok.UnkId);
        Assert.Equal(4, tok.MaskId);
    }

    [Fact]
    public void NoTokenizerConfig_NoRecognisedSpelling_ThrowsNamingTheRole()
    {
        // A vocabulary that uses none of the conventional spellings for cls_token must fail
        // loudly and name which role could not be resolved, rather than silently picking an
        // unrelated id or crashing somewhere downstream with no context.
        var vocab = """{"[PAD]":0,"[UNK]":1,"[SEP]":3,"[MASK]":4,"hello":5}""";
        var added = string.Join(",", [
            AddedToken("[PAD]", 0), AddedToken("[UNK]", 1),
            AddedToken("[SEP]", 3), AddedToken("[MASK]", 4),
        ]);
        WriteTokenizerJson(added, vocab);

        var ex = Assert.Throws<InvalidOperationException>(
            () => new HfTokenizer(Path.Combine(_dir, "tokenizer.json")));

        Assert.Contains("cls_token", ex.Message);
    }

    [Fact]
    public void TokenizerConfigPresent_UsesItInsteadOfGuessing()
    {
        // Fused layout: tokenizer_config.json is present, so the config-driven path (not the
        // fallback) must be the one that runs, even though the fallback's conventional spellings
        // would also happen to resolve on this vocabulary.
        var added = string.Join(",", [
            AddedToken("[PAD]", 0), AddedToken("[UNK]", 1), AddedToken("[CLS]", 2),
            AddedToken("[SEP]", 3), AddedToken("[MASK]", 4),
        ]);
        WriteTokenizerJson(added, BracketedVocab);
        File.WriteAllText(Path.Combine(_dir, "tokenizer_config.json"), """
            {
              "cls_token": "[CLS]",
              "sep_token": "[SEP]",
              "pad_token": "[PAD]",
              "unk_token": "[UNK]",
              "mask_token": "[MASK]"
            }
            """);

        using var tok = new HfTokenizer(Path.Combine(_dir, "tokenizer.json"));

        Assert.Equal(0, tok.PadId);
        Assert.Equal(2, tok.ClsId);
        Assert.Equal(3, tok.SepId);
        Assert.Equal(4, tok.MaskId);
        Assert.Equal(1, tok.UnkId);
    }
}
