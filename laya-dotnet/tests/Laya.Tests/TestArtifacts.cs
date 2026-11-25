using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Resolved artifact paths for a single checkpoint, so tests that need them can be skipped
/// rather than failing on a machine that has not downloaded the checkpoint.
/// </summary>
public sealed class CheckpointTestArtifacts
{
    private readonly LayaCheckpoint _checkpoint;
    private readonly Lazy<string?> _directoryLazy;
    private readonly Lazy<string?> _splitDirectoryLazy;

    internal CheckpointTestArtifacts(LayaCheckpoint checkpoint)
    {
        _checkpoint = checkpoint;
        _directoryLazy = new(Locate);
        _splitDirectoryLazy = new(() => LocateSplitFor(checkpoint));
    }

    /// <summary>
    /// The artifact directory for this checkpoint, or <see langword="null"/> if none was found.
    /// </summary>
    public string? DirectoryOrNull => _directoryLazy.Value;

    /// <summary>The artifact directory. Throws <see cref="DirectoryNotFoundException"/> if absent.</summary>
    public string Directory => DirectoryOrNull
        ?? throw new DirectoryNotFoundException(
            $"no ONNX artifact directory found for '{_checkpoint}'; "
            + "set LAYA_ONNX_ROOT to the parent of all checkpoints, or LAYA_ONNX_DIR for multilingual");

    /// <summary><c>tokenizer/tokenizer.json</c> is present.</summary>
    public bool HasTokenizer =>
        DirectoryOrNull is { } d && File.Exists(Path.Combine(d, "tokenizer", "tokenizer.json"));

    /// <summary>The graph and its external-data sidecar are both present.</summary>
    public bool HasModel =>
        DirectoryOrNull is { } d
        && File.Exists(Path.Combine(d, "model.onnx"))
        && File.Exists(Path.Combine(d, "model.onnx.data"));

    /// <summary>The tokenizer path, or a skip when it is missing.</summary>
    public string RequireTokenizer()
    {
        if (!HasTokenizer)
            Assert.Skip($"no tokenizer.json for '{_checkpoint}'; set LAYA_ONNX_ROOT or LAYA_ONNX_DIR");
        return Path.Combine(DirectoryOrNull!, "tokenizer", "tokenizer.json");
    }

    /// <summary>The artifact directory, or a skip when the model is missing.</summary>
    public string RequireModel()
    {
        if (!HasModel)
            Assert.Skip($"no model.onnx for '{_checkpoint}'; set LAYA_ONNX_ROOT or LAYA_ONNX_DIR");
        return DirectoryOrNull!;
    }

    // ── split layout (laya-ts exporter output) ─────────────────────────────────

    /// <summary>
    /// The split-layout artifact directory for this checkpoint, or <see langword="null"/> if
    /// none was found. Looked up separately from <see cref="DirectoryOrNull"/> because a
    /// checkpoint can have a fused export, a split export, both, or neither on a given machine.
    /// </summary>
    public string? SplitDirectoryOrNull => _splitDirectoryLazy.Value;

    /// <summary><c>encoder.onnx</c> + <c>head.onnx</c> (+ <c>rl_agent_config.json</c> + <c>tokenizer.json</c>) are all present.</summary>
    public bool HasSplitModel =>
        SplitDirectoryOrNull is { } d
        && File.Exists(Path.Combine(d, "encoder.onnx"))
        && File.Exists(Path.Combine(d, "head.onnx"))
        && File.Exists(Path.Combine(d, "tokenizer.json"));

    /// <summary>The split artifact directory, or a skip when it is missing.</summary>
    public string RequireSplitModel()
    {
        if (!HasSplitModel)
            Assert.Skip(
                $"no split ONNX artifact for '{_checkpoint}'; export via laya-ts's exporter into "
                + $"<repo>/onnx-split/{LayaOptions.CheckpointSubfolder(_checkpoint)}");
        return SplitDirectoryOrNull!;
    }

    // ── resolution ─────────────────────────────────────────────────────────────

    private string? Locate()
    {
        var subfolder = LayaOptions.CheckpointSubfolder(_checkpoint);

        // LAYA_ONNX_DIR is the original single-checkpoint env var that always pointed at the
        // multilingual artifact directory. It must keep doing so for backward compat; new code
        // should use LAYA_ONNX_ROOT.
        if (_checkpoint == LayaCheckpoint.Multilingual)
        {
            var envDir = Environment.GetEnvironmentVariable("LAYA_ONNX_DIR");
            if (!string.IsNullOrWhiteSpace(envDir))
            {
                // A set variable wins outright, even when it points nowhere. Falling back would
                // silently test a different directory than the one the user specified, and would
                // make the skip path unreachable on a machine that also has a local export.
                return System.IO.Directory.Exists(envDir) ? Path.GetFullPath(envDir) : null;
            }
        }

        // LAYA_ONNX_ROOT is the new multi-checkpoint root: each checkpoint lives in a named
        // subdirectory. If the root is set but this checkpoint's subdir does not exist, fall
        // through to the walk-up so a developer who has only one checkpoint under the root can
        // still run the matching tests.
        var onnxRoot = Environment.GetEnvironmentVariable("LAYA_ONNX_ROOT");
        if (!string.IsNullOrWhiteSpace(onnxRoot))
        {
            var rootDir = Path.Combine(onnxRoot, subfolder);
            if (System.IO.Directory.Exists(rootDir)
                && File.Exists(Path.Combine(rootDir, "rl_agent_config.json")))
                return Path.GetFullPath(rootDir);
        }

        // Walk up from the test binary to the repository root and look for onnx/<subfolder>.
        // A developer who has exported locally gets the full suite without setting anything.
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "onnx", subfolder);
            if (File.Exists(Path.Combine(candidate, "rl_agent_config.json")))
                return candidate;
        }
        return null;
    }

    private static string? LocateSplitFor(LayaCheckpoint checkpoint)
    {
        var subfolder = LayaOptions.CheckpointSubfolder(checkpoint);

        // There is no split-layout equivalent of LAYA_ONNX_DIR/LAYA_ONNX_ROOT yet (the split
        // layout is new); only the walk-up from the test binary to <repo>/onnx-split/<subfolder>
        // is supported, mirroring the fused layout's walk-up fallback.
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "onnx-split", subfolder);
            if (File.Exists(Path.Combine(candidate, "encoder.onnx")))
                return candidate;
        }
        return null;
    }
}

/// <summary>
/// Finds ONNX artifacts for each checkpoint, so tests that need them can be skipped rather than
/// failing on a machine that has not downloaded the checkpoints.
/// </summary>
/// <remarks>
/// The tokenizer and the model are tracked separately: <c>tokenizer.json</c> alone is enough to
/// check token-level parity, which is where a port like this is most likely to be wrong, and it is
/// a much smaller prerequisite than the weights.
/// </remarks>
public static class TestArtifacts
{
    /// <summary>Checkpoints selected for a parity job; ordinary local runs still cover all three.</summary>
    public static IEnumerable<LayaCheckpoint> ParityCheckpoints
    {
        get
        {
            var selected = Environment.GetEnvironmentVariable("LAYA_TEST_CHECKPOINTS");
            var values = Enum.GetValues<LayaCheckpoint>();
            if (string.IsNullOrWhiteSpace(selected))
                return values;
            var names = selected.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (names.Any(name => !values.Any(c => LayaOptions.CheckpointSubfolder(c) == name)))
                throw new ArgumentException($"unknown LAYA_TEST_CHECKPOINTS: {selected}");
            return values.Where(c => names.Contains(LayaOptions.CheckpointSubfolder(c)));
        }
    }

    /// <summary>
    /// The trait that places a layout-specific test in a CI cell: <c>Layout=split-only</c> runs
    /// against the split export only, <c>Layout=any</c> is parameterised by layout (its rows are
    /// narrowed by <see cref="ParitySplitFlags"/>), and an untagged test is fused or layout
    /// independent. A split cell runs <c>split-only</c> and <c>any</c>; a fused cell runs everything
    /// except <c>split-only</c>. Filtering by trait (rather than skipping at run time) means a
    /// deselected layout produces no skipped test.
    /// </summary>
    public const string LayoutTrait = "Layout";

    /// <summary>Value of <see cref="LayoutTrait"/> for a test that only exercises the split layout.</summary>
    public const string SplitOnly = "split-only";

    /// <summary>Value of <see cref="LayoutTrait"/> for a theory with a fused and a split row per case.</summary>
    public const string AnyLayout = "any";

    /// <summary>
    /// The <c>split</c> flag values a layout-parameterised theory yields (<c>false</c> is the fused
    /// layout, <c>true</c> the split one). <c>LAYA_TEST_LAYOUTS</c> is a space-separated list of
    /// <c>fused</c> and <c>split</c>; unset selects both, so ordinary local runs are unchanged.
    /// </summary>
    public static IEnumerable<bool> ParitySplitFlags
    {
        get
        {
            var selected = Environment.GetEnvironmentVariable("LAYA_TEST_LAYOUTS");
            if (string.IsNullOrWhiteSpace(selected))
                return [false, true];
            var names = selected.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (names.Any(name => name != "fused" && name != "split"))
                throw new ArgumentException($"unknown LAYA_TEST_LAYOUTS: {selected}");
            return new[] { false, true }.Where(split => names.Contains(split ? "split" : "fused"));
        }
    }

    // One CheckpointTestArtifacts per enum value; indexed by the enum's integer value.
    private static readonly CheckpointTestArtifacts[] ByCheckpoint =
        Enum.GetValues<LayaCheckpoint>().Select(c => new CheckpointTestArtifacts(c)).ToArray();

    /// <summary>Returns per-checkpoint artifact access for <paramref name="checkpoint"/>.</summary>
    public static CheckpointTestArtifacts For(LayaCheckpoint checkpoint) =>
        ByCheckpoint[(int)checkpoint];

    // ── backward-compat shims pointing at the multilingual checkpoint ─────────

    /// <summary>The multilingual artifact directory, or null if none was found.</summary>
    public static string? Directory => For(LayaCheckpoint.Multilingual).DirectoryOrNull;

    /// <summary><c>tokenizer/tokenizer.json</c> is present for the multilingual checkpoint.</summary>
    public static bool HasTokenizer => For(LayaCheckpoint.Multilingual).HasTokenizer;

    /// <summary>The graph and sidecar are both present for the multilingual checkpoint.</summary>
    public static bool HasModel => For(LayaCheckpoint.Multilingual).HasModel;

    /// <summary>The multilingual tokenizer path, or a skip if it is missing.</summary>
    public static string RequireTokenizer() => For(LayaCheckpoint.Multilingual).RequireTokenizer();

    /// <summary>The multilingual artifact directory, or a skip if the model is missing.</summary>
    public static string RequireModel() => For(LayaCheckpoint.Multilingual).RequireModel();
}

/// <summary>
/// One <see cref="LayaEngine"/> shared by every test that needs inference, loaded per checkpoint.
/// </summary>
/// <remarks>
/// Loading the graph and its 1.29 GB of weights takes seconds and hundreds of megabytes, and the
/// engine is documented as safe to share, so the suite builds at most one engine per checkpoint and
/// thereby also exercises that claim.
/// </remarks>
public sealed class AllEnginesFixture : IDisposable
{
    // One lazy engine per LayaCheckpoint value, indexed by the enum's integer value.
    private readonly Lazy<LayaEngine>[] _engines;

    // Same, but loaded from the split-layout artifact directory (<repo>/onnx-split/<ckpt>), when
    // present. Kept in the same fixture as the fused engines rather than a separate fixture class:
    // both are cheap until first .Value access, and sharing one collection fixture means a test
    // class that wants both layouts does not need two [Collection] attributes.
    private readonly Lazy<LayaEngine>[] _splitEngines;

    // The same two layouts loaded with the calibration file the `calibration_*` golden cases were recorded with.
    // A file replaces the checkpoint's temperatures for every question, so those cases cannot share the engines above.
    private readonly Lazy<LayaEngine>[] _calibratedEngines;
    private readonly Lazy<LayaEngine>[] _calibratedSplitEngines;

    /// <summary>Create the fixture. Engines are not loaded until first use.</summary>
    public AllEnginesFixture()
    {
        var values = Enum.GetValues<LayaCheckpoint>();
        _engines = new Lazy<LayaEngine>[values.Length];
        _splitEngines = new Lazy<LayaEngine>[values.Length];
        _calibratedEngines = new Lazy<LayaEngine>[values.Length];
        _calibratedSplitEngines = new Lazy<LayaEngine>[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            var checkpoint = values[i]; // captured correctly: new variable each iteration
            // The recorded language cases name a table of per-language temperatures; an engine that has it answers
            // every other case as one without (a language is only looked up when a call names one).
            _engines[(int)checkpoint] = new Lazy<LayaEngine>(() => LayaEngine.Create(new LayaOptions
            {
                ModelDirectory = TestArtifacts.For(checkpoint).Directory,
                LangTemperatures = GoldenData.For(checkpoint).LangTemperatures,
            }));
            _splitEngines[(int)checkpoint] = new Lazy<LayaEngine>(() => LayaEngine.Create(new LayaOptions
            {
                ModelDirectory = TestArtifacts.For(checkpoint).SplitDirectoryOrNull!,
                LangTemperatures = GoldenData.For(checkpoint).LangTemperatures,
            }));
            _calibratedEngines[(int)checkpoint] = new Lazy<LayaEngine>(() => LayaEngine.Create(new LayaOptions
            {
                ModelDirectory = TestArtifacts.For(checkpoint).Directory,
                LangTemperatures = GoldenData.For(checkpoint).LangTemperatures,
                Calibration = CalibrationFixture.Path,
            }));
            _calibratedSplitEngines[(int)checkpoint] = new Lazy<LayaEngine>(() => LayaEngine.Create(new LayaOptions
            {
                ModelDirectory = TestArtifacts.For(checkpoint).SplitDirectoryOrNull!,
                LangTemperatures = GoldenData.For(checkpoint).LangTemperatures,
                Calibration = CalibrationFixture.Path,
            }));
        }
    }

    /// <summary>Whether an engine for <paramref name="checkpoint"/> can be loaded.</summary>
    public bool Available(LayaCheckpoint checkpoint) =>
        TestArtifacts.For(checkpoint).HasModel && TestArtifacts.For(checkpoint).HasTokenizer;

    /// <summary>Whether a split-layout engine for <paramref name="checkpoint"/> can be loaded.</summary>
    public bool AvailableSplit(LayaCheckpoint checkpoint) =>
        TestArtifacts.For(checkpoint).HasSplitModel;

    /// <summary>The shared engine for <paramref name="checkpoint"/>, or a skip when artifacts are missing.</summary>
    public LayaEngine Require(LayaCheckpoint checkpoint)
    {
        if (!Available(checkpoint))
            Assert.Skip(
                $"no ONNX artifacts for '{checkpoint}'; set LAYA_ONNX_ROOT to the parent of all " +
                "checkpoints, or LAYA_ONNX_DIR for multilingual");
        return _engines[(int)checkpoint].Value;
    }

    /// <summary>
    /// The shared split-layout engine for <paramref name="checkpoint"/>, or a skip when the
    /// split artifact (<c>&lt;repo&gt;/onnx-split/&lt;checkpoint&gt;</c>) is missing.
    /// </summary>
    public LayaEngine RequireSplit(LayaCheckpoint checkpoint)
    {
        if (!AvailableSplit(checkpoint))
            Assert.Skip(
                $"no split ONNX artifact for '{checkpoint}'; export it via laya-ts's exporter into "
                + $"<repo>/onnx-split/{LayaOptions.CheckpointSubfolder(checkpoint)}");
        return _splitEngines[(int)checkpoint].Value;
    }

    /// <summary>
    /// The engine a recorded case was answered by: the one loaded with the calibration file when the case was recorded with it (after
    /// checking that the file is still the one recorded), else the shared engine of the layout.
    /// </summary>
    internal LayaEngine RequireFor(LayaCheckpoint checkpoint, JsonElement root, bool split = false)
    {
        if (!CalibrationFixture.Recorded(root)) return split ? RequireSplit(checkpoint) : Require(checkpoint);
        CalibrationFixture.AssertIsTheRecordedFile(root);
        if (split)
        {
            RequireSplit(checkpoint);
            return _calibratedSplitEngines[(int)checkpoint].Value;
        }
        Require(checkpoint);
        return _calibratedEngines[(int)checkpoint].Value;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var lazy in _calibratedEngines)
            if (lazy.IsValueCreated) lazy.Value.Dispose();
        foreach (var lazy in _calibratedSplitEngines)
            if (lazy.IsValueCreated) lazy.Value.Dispose();
        foreach (var lazy in _engines)
            if (lazy.IsValueCreated) lazy.Value.Dispose();
        foreach (var lazy in _splitEngines)
            if (lazy.IsValueCreated) lazy.Value.Dispose();
    }
}

/// <summary>Shares one <see cref="AllEnginesFixture"/> across every member of the collection.</summary>
[CollectionDefinition(Name)]
public sealed class AllEnginesCollection : ICollectionFixture<AllEnginesFixture>
{
    /// <summary>The collection name.</summary>
    public const string Name = "all-engines";
}

/// <summary>
/// One <see cref="LayaEngine"/> shared by every test that needs multilingual inference.
/// </summary>
/// <remarks>
/// Kept for backward compatibility. New tests should use <see cref="AllEnginesFixture"/> to cover
/// all checkpoints.
/// </remarks>
public sealed class EngineFixture : IDisposable
{
    private readonly Lazy<LayaEngine> _engine;

    /// <summary>Create the fixture. The engine itself is not loaded until first use.</summary>
    public EngineFixture() =>
        _engine = new(() => LayaEngine.FromDirectory(TestArtifacts.Directory!));

    /// <summary>Whether an engine can be loaded at all.</summary>
    public bool Available => TestArtifacts.HasModel && TestArtifacts.HasTokenizer;

    /// <summary>The shared engine, or a skip when the artifacts are missing.</summary>
    public LayaEngine Require()
    {
        if (!Available)
            Assert.Skip("no ONNX artifacts found; set LAYA_ONNX_DIR to an artifact directory");
        return _engine.Value;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_engine.IsValueCreated) _engine.Value.Dispose();
    }
}

/// <summary>Shares one <see cref="EngineFixture"/> across every collection member.</summary>
[CollectionDefinition(Name)]
public sealed class EngineCollection : ICollectionFixture<EngineFixture>
{
    /// <summary>The collection name.</summary>
    public const string Name = "laya-engine";
}
