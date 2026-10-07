namespace Laya.Tests;

/// <summary>
/// Tier 1: layout detection and tokenizer-path lookup, using empty placeholder files in a temp
/// directory rather than real weights. These pin the branching in
/// <see cref="ModelArtifacts"/> that decides fused vs. split (and reports both layouts when
/// neither is found) without needing anything downloaded.
/// </summary>
public sealed class ModelArtifactsTests : IDisposable
{
    private readonly string _dir;

    public ModelArtifactsTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "laya-artifact-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); }
        catch (IOException) { /* best effort */ }
    }

    private void Touch(params string[] relativeSegments)
    {
        var path = Path.Combine(_dir, Path.Combine(relativeSegments));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, []);
    }

    private ModelArtifacts Resolve() =>
        ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = _dir });

    [Fact]
    public void FusedOnly_IsDetectedAsFused()
    {
        Touch("model.onnx");
        Touch("model.onnx.data");
        Touch("rl_agent_config.json");
        Touch("tokenizer", "tokenizer.json");

        var artifacts = Resolve();

        Assert.Equal(LayaArtifactLayout.Fused, artifacts.Layout);
        Assert.NotNull(artifacts.ModelPath);
        Assert.Null(artifacts.EncoderPath);
        Assert.Null(artifacts.HeadPath);
    }

    [Fact]
    public void SplitOnly_IsDetectedAsSplit()
    {
        Touch("encoder.onnx");
        Touch("head.onnx");
        Touch("rl_agent_config.json");
        Touch("tokenizer.json");

        var artifacts = Resolve();

        Assert.Equal(LayaArtifactLayout.Split, artifacts.Layout);
        Assert.Null(artifacts.ModelPath);
        Assert.NotNull(artifacts.EncoderPath);
        Assert.NotNull(artifacts.HeadPath);
    }

    [Fact]
    public void BothLayoutsPresent_FusedWins()
    {
        // A directory that somehow has both (e.g. a developer exported one on top of the other)
        // must not become ambiguous: the fused layout is what this SDK has always produced, so
        // it is the one that wins.
        Touch("model.onnx");
        Touch("model.onnx.data");
        Touch("encoder.onnx");
        Touch("head.onnx");
        Touch("rl_agent_config.json");
        Touch("tokenizer", "tokenizer.json");

        var artifacts = Resolve();

        Assert.Equal(LayaArtifactLayout.Fused, artifacts.Layout);
    }

    [Fact]
    public void NeitherLayoutPresent_ThrowsListingBothLayouts()
    {
        Touch("rl_agent_config.json");

        var ex = Assert.Throws<FileNotFoundException>(Resolve);

        Assert.Contains("model.onnx", ex.Message);
        Assert.Contains("encoder.onnx", ex.Message);
        Assert.Contains("head.onnx", ex.Message);
    }

    [Fact]
    public void HalfSplit_EncoderWithoutHead_ThrowsListingBothLayouts()
    {
        // Only encoder.onnx: an interrupted or partial split export must fall through to the
        // same "neither layout matched" error as a genuinely empty directory, not silently pick
        // the split branch with a null head session.
        Touch("encoder.onnx");
        Touch("rl_agent_config.json");
        Touch("tokenizer.json");

        var ex = Assert.Throws<FileNotFoundException>(Resolve);

        Assert.Contains("model.onnx", ex.Message);
        Assert.Contains("encoder.onnx", ex.Message);
        Assert.Contains("head.onnx", ex.Message);
    }

    [Fact]
    public void HalfSplit_HeadWithoutEncoder_ThrowsListingBothLayouts()
    {
        Touch("head.onnx");
        Touch("rl_agent_config.json");
        Touch("tokenizer.json");

        var ex = Assert.Throws<FileNotFoundException>(Resolve);

        Assert.Contains("model.onnx", ex.Message);
        Assert.Contains("encoder.onnx", ex.Message);
        Assert.Contains("head.onnx", ex.Message);
    }

    [Fact]
    public void TokenizerLookup_PrefersNestedTokenizerDirectoryOverRoot()
    {
        // Both locations exist (a directory that has both a fused-style tokenizer/ subfolder and
        // a root-level tokenizer.json, however that came about): the nested one must win, since
        // that is where the fused layout's downloader and every fused artifact puts it.
        Touch("model.onnx");
        Touch("model.onnx.data");
        Touch("rl_agent_config.json");
        Touch("tokenizer", "tokenizer.json");
        Touch("tokenizer.json");

        var artifacts = Resolve();

        Assert.Equal(Path.Combine(_dir, "tokenizer", "tokenizer.json"), artifacts.TokenizerPath);
    }

    [Fact]
    public void TokenizerLookup_FallsBackToRootWhenNoNestedDirectory()
    {
        // The split layout's tokenizer.json sits directly in the artifact directory, with no
        // tokenizer/ subfolder at all.
        Touch("encoder.onnx");
        Touch("head.onnx");
        Touch("rl_agent_config.json");
        Touch("tokenizer.json");

        var artifacts = Resolve();

        Assert.Equal(Path.Combine(_dir, "tokenizer.json"), artifacts.TokenizerPath);
    }

    [Fact]
    public void MissingTokenizer_ThrowsNamingBothLocationsTried()
    {
        Touch("model.onnx");
        Touch("model.onnx.data");
        Touch("rl_agent_config.json");

        var ex = Assert.Throws<FileNotFoundException>(Resolve);

        Assert.Contains(Path.Combine(_dir, "tokenizer", "tokenizer.json"), ex.Message);
        Assert.Contains(Path.Combine(_dir, "tokenizer.json"), ex.Message);
    }

    [Fact]
    public void MissingModelDotOnnxData_ThrowsNamingTheSidecar()
    {
        // model.onnx alone (no .data sidecar) must fail with a message pointing at the sidecar
        // specifically, not the generic "neither layout" message — a half-fused export is a much
        // more common mistake (a copy that missed the large external-data file) than a half-split
        // one, and deserves its own hint.
        Touch("model.onnx");
        Touch("rl_agent_config.json");
        Touch("tokenizer", "tokenizer.json");

        var ex = Assert.Throws<FileNotFoundException>(Resolve);

        Assert.Contains("model.onnx.data", ex.Message);
    }

    [Fact]
    public void MissingDirectory_ThrowsDirectoryNotFound()
    {
        var missing = Path.Combine(_dir, "does-not-exist");
        Assert.Throws<DirectoryNotFoundException>(
            () => ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = missing }));
    }
}
