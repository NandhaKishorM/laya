namespace Laya;

/// <summary>
/// Selects which Laya checkpoint to load. Each maps to a different encoder and context length;
/// see the SDK README for the tradeoffs.
/// </summary>
public enum LayaCheckpoint
{
    /// <summary>
    /// mmBERT-base (322 M), 1 024-token context, 100+ languages. The safest default for
    /// mixed-language traffic; covers English too, so there is no router to configure.
    /// </summary>
    Multilingual,

    /// <summary>
    /// ModernBERT-large (421 M), 512-token context, English-optimised. Higher accuracy on
    /// English-only workloads; accuracy collapses on non-Latin scripts (route those through
    /// the multilingual checkpoint).
    /// </summary>
    English,

    /// <summary>
    /// ModernBERT-large (421 M), 1 024-token context, fine-tuned on four typed-decisions
    /// workflows. Use only for the specific workflows it was trained on.
    /// </summary>
    TypedDecisions,
}

/// <summary>
/// ONNX execution provider. Only <see cref="Cpu"/> is wired in v1; the others are reserved so
/// callers can opt in and receive a clear error if the matching ORT package is absent.
/// </summary>
public enum LayaExecutionProvider
{
    /// <summary>CPU execution (default).</summary>
    Cpu,

    /// <summary>NVIDIA CUDA. Requires the <c>Microsoft.ML.OnnxRuntime.Gpu</c> package.</summary>
    Cuda,

    /// <summary>Windows DirectML. Requires the <c>Microsoft.ML.OnnxRuntime.DirectML</c> package.</summary>
    DirectMl,
}

/// <summary>Progress report for a single file being downloaded from HuggingFace Hub.</summary>
/// <param name="FileName">The file name being fetched, e.g. <c>model.onnx.data</c>.</param>
/// <param name="BytesReceived">Bytes written to disk so far.</param>
/// <param name="TotalBytes">
/// Total file size in bytes, or <see langword="null"/> when the server omits
/// <c>Content-Length</c>.
/// </param>
public readonly record struct ArtifactDownloadProgress(
    string FileName,
    long BytesReceived,
    long? TotalBytes);

/// <summary>
/// Controls how the Laya engine locates or downloads its ONNX artifact and how it
/// configures ONNX Runtime.
/// </summary>
public sealed class LayaOptions
{
    private string? _huggingFaceSubfolder;

    /// <summary>
    /// The directory that contains <c>model.onnx</c>. When set, skips all other
    /// resolution steps. Default <see langword="null"/> (auto-resolve).
    /// </summary>
    public string? ModelDirectory { get; set; }

    /// <summary>
    /// Which checkpoint to load. Default <see cref="LayaCheckpoint.Multilingual"/>.
    /// The subdirectory, HuggingFace subfolder, and cache path all derive from this value
    /// unless overridden by <see cref="HuggingFaceSubfolder"/> or <see cref="ModelDirectory"/>.
    /// </summary>
    public LayaCheckpoint Checkpoint { get; set; } = LayaCheckpoint.Multilingual;

    /// <summary>
    /// Which hardware back-end to use. Default <see cref="LayaExecutionProvider.Cpu"/>.
    /// </summary>
    public LayaExecutionProvider ExecutionProvider { get; set; } = LayaExecutionProvider.Cpu;

    /// <summary>
    /// Number of threads used within a single ONNX operator.
    /// <see langword="null"/> lets ONNX Runtime choose (usually one per logical CPU).
    /// </summary>
    public int? IntraOpThreads { get; set; }

    /// <summary>
    /// Number of threads used to run independent ONNX operators in parallel.
    /// <see langword="null"/> lets ONNX Runtime choose.
    /// </summary>
    public int? InterOpThreads { get; set; }

    /// <summary>
    /// Whether the engine may fetch the artifact from HuggingFace Hub when it is not
    /// found locally. Default <see langword="false"/>; downloading must always be opt-in.
    /// </summary>
    public bool AllowDownload { get; set; } = false;

    /// <summary>
    /// HuggingFace repository that hosts the artifact.
    /// Default <c>"convaiinnovations/laya"</c>.
    /// </summary>
    public string HuggingFaceRepo { get; set; } = "convaiinnovations/laya";

    /// <summary>
    /// Subfolder within the repository and local cache. Derived from <see cref="Checkpoint"/>
    /// by default. Set explicitly only when loading a custom export that does not follow the
    /// standard bundle layout — prefer setting <see cref="Checkpoint"/> instead.
    /// </summary>
    /// <remarks>
    /// Note that the HuggingFace bundle subfolder for the English checkpoint is the bundle
    /// root (empty string), not "english". When this property is left at its default,
    /// download URLs are computed correctly. If you set it explicitly for English you must
    /// use the empty string for the download path; for local artifact directories "english"
    /// is the conventional name.
    /// </remarks>
    public string HuggingFaceSubfolder
    {
        get => _huggingFaceSubfolder ?? CheckpointSubfolder(Checkpoint);
        set => _huggingFaceSubfolder = value;
    }

    /// <summary>
    /// HuggingFace access token used in the <c>Authorization: Bearer</c> header.
    /// When <see langword="null"/>, falls back to the <c>HF_TOKEN</c> environment variable.
    /// Set to an empty string to suppress all auth (public repos only).
    /// </summary>
    public string? HuggingFaceToken { get; set; }

    /// <summary>
    /// Root directory under which the downloaded artifact is stored. Default
    /// <see langword="null"/>, which resolves to
    /// <c>%LOCALAPPDATA%\laya\onnx</c> on Windows and <c>~/.cache/laya/onnx</c>
    /// elsewhere.
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Optional sink for per-file download progress. Each notification carries the
    /// file name, bytes received so far, and total size when known.
    /// </summary>
    public IProgress<ArtifactDownloadProgress>? DownloadProgress { get; set; }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// <summary>
    /// The local artifact subdirectory name for <paramref name="c"/>.
    /// Used as the subdirectory inside the cache root and inside an <c>onnx/</c> tree.
    /// </summary>
    internal static string CheckpointSubfolder(LayaCheckpoint c) => c switch
    {
        LayaCheckpoint.Multilingual   => "multilingual",
        LayaCheckpoint.English        => "english",
        LayaCheckpoint.TypedDecisions => "typed-decisions",
        _ => throw new ArgumentOutOfRangeException(nameof(c), c, null),
    };

    /// <summary>
    /// The subfolder path to use in HuggingFace download URLs. For English the checkpoint
    /// lives at the bundle root (empty string), which differs from the local directory name
    /// "english". When <see cref="HuggingFaceSubfolder"/> was set explicitly that value is
    /// used as-is.
    /// </summary>
    internal string HuggingFaceDownloadSubfolder =>
        _huggingFaceSubfolder ?? HfDownloadSubfolder(Checkpoint);

    private static string HfDownloadSubfolder(LayaCheckpoint c) => c switch
    {
        LayaCheckpoint.Multilingual   => "multilingual",
        LayaCheckpoint.English        => "",             // lives at the bundle root
        LayaCheckpoint.TypedDecisions => "typed-decisions",
        _ => throw new ArgumentOutOfRangeException(nameof(c), c, null),
    };
}
