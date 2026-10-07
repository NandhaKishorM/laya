using System.Net.Http.Headers;

namespace Laya;

/// <summary>
/// Which of the two ONNX artifact shapes an artifact directory contains.
/// </summary>
public enum LayaArtifactLayout
{
    /// <summary><c>model.onnx</c> + <c>model.onnx.data</c>, a single graph for the whole forward pass.</summary>
    Fused,

    /// <summary>
    /// <c>encoder.onnx</c> + <c>head.onnx</c> (each optionally with a <c>.data</c> sidecar), the
    /// layout produced by laya-ts's exporter (<c>laya-ts/scripts/export_onnx.py</c>).
    /// </summary>
    Split,
}

/// <summary>
/// Resolved, validated paths to every file the Laya engine needs at startup.
/// Obtain via <see cref="Resolve"/> or <see cref="ResolveAsync"/>.
/// </summary>
public sealed class ModelArtifacts
{
    /// <summary>The directory that contains the artifact.</summary>
    public string Directory { get; }

    /// <summary>Which layout was found in <see cref="Directory"/>.</summary>
    public LayaArtifactLayout Layout { get; }

    /// <summary>Absolute path to <c>model.onnx</c>, or <see langword="null"/> for <see cref="LayaArtifactLayout.Split"/>.</summary>
    public string? ModelPath { get; }

    /// <summary>Absolute path to <c>encoder.onnx</c>, or <see langword="null"/> for <see cref="LayaArtifactLayout.Fused"/>.</summary>
    public string? EncoderPath { get; }

    /// <summary>Absolute path to <c>head.onnx</c>, or <see langword="null"/> for <see cref="LayaArtifactLayout.Fused"/>.</summary>
    public string? HeadPath { get; }

    /// <summary>Absolute path to <c>rl_agent_config.json</c>.</summary>
    public string ConfigPath { get; }

    /// <summary>
    /// Absolute path to <c>tokenizer.json</c> — under <c>tokenizer/</c> for the fused layout,
    /// or (usually) directly in <see cref="Directory"/> for the split layout. See
    /// <see cref="ResolveTokenizerPath"/> for the exact lookup order.
    /// </summary>
    public string TokenizerPath { get; }

    /// <summary>
    /// The checkpoint this artifact directory corresponds to, or <see langword="null"/>
    /// when loaded from an explicit path (via <see cref="LayaOptions.ModelDirectory"/> or
    /// the <c>LAYA_ONNX_DIR</c> environment variable), in which case no encoder validation
    /// is performed.
    /// </summary>
    public LayaCheckpoint? Checkpoint { get; }

    private ModelArtifacts(string directory, LayaArtifactLayout layout, string? modelPath,
                            string? encoderPath, string? headPath, string configPath,
                            string tokenizerPath, LayaCheckpoint? checkpoint)
    {
        Directory      = directory;
        Layout         = layout;
        ModelPath      = modelPath;
        EncoderPath    = encoderPath;
        HeadPath       = headPath;
        ConfigPath     = configPath;
        TokenizerPath  = tokenizerPath;
        Checkpoint     = checkpoint;
    }

    /// <summary>
    /// Resolves and validates the artifact directory synchronously.
    /// Equivalent to calling <see cref="ResolveAsync"/> and blocking on the result.
    /// </summary>
    /// <exception cref="DirectoryNotFoundException">
    /// When no artifact is found and <see cref="LayaOptions.AllowDownload"/> is
    /// <see langword="false"/>.
    /// </exception>
    /// <exception cref="FileNotFoundException">
    /// When the resolved directory is missing a required file.
    /// </exception>
    public static ModelArtifacts Resolve(LayaOptions options)
        => ResolveAsync(options).GetAwaiter().GetResult();

    /// <summary>
    /// Resolves and validates the artifact directory. Downloads the artifact when
    /// <see cref="LayaOptions.AllowDownload"/> is <see langword="true"/> and it is not
    /// already present.
    /// </summary>
    /// <remarks>
    /// Resolution order:
    /// <list type="number">
    ///   <item><see cref="LayaOptions.ModelDirectory"/> if set.</item>
    ///   <item>The <c>LAYA_ONNX_DIR</c> environment variable if set (backward compat;
    ///   always treated as pointing directly at the artifact directory).</item>
    ///   <item>The <c>LAYA_ONNX_ROOT</c> environment variable combined with the
    ///   checkpoint subdirectory name, e.g. <c>$LAYA_ONNX_ROOT/multilingual</c>.</item>
    ///   <item>The cache directory (<c>&lt;CacheDirectory&gt;/&lt;subfolder&gt;</c>) if it
    ///   already contains a complete artifact.</item>
    ///   <item>Download into the cache if <see cref="LayaOptions.AllowDownload"/> is
    ///   <see langword="true"/>; otherwise throw.</item>
    /// </list>
    /// Every step accepts either the fused or the split layout; see
    /// <see cref="LayaArtifactLayout"/>. The fused layout wins when a directory somehow has
    /// both, since it is the one this SDK has always produced.
    /// </remarks>
    /// <exception cref="DirectoryNotFoundException">
    /// When no artifact is found and <see cref="LayaOptions.AllowDownload"/> is
    /// <see langword="false"/>.
    /// </exception>
    /// <exception cref="FileNotFoundException">
    /// When the resolved directory is missing a required file.
    /// </exception>
    public static async Task<ModelArtifacts> ResolveAsync(
        LayaOptions options, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        // 1. Explicit directory. No checkpoint inference: the caller knows what they loaded.
        if (!string.IsNullOrEmpty(options.ModelDirectory))
            return ValidateDirectory(options.ModelDirectory, checkpoint: null);

        // 2. LAYA_ONNX_DIR: backward-compat env var that points directly at the artifact dir.
        //    Treated as an explicit path for the same reason: it may point at any checkpoint.
        var envDir = Environment.GetEnvironmentVariable("LAYA_ONNX_DIR");
        if (!string.IsNullOrEmpty(envDir))
            return ValidateDirectory(envDir, checkpoint: null);

        var subfolder = LayaOptions.CheckpointSubfolder(options.Checkpoint);

        // 3. LAYA_ONNX_ROOT: new env var for a tree that holds every checkpoint in named
        //    subdirectories (english/, multilingual/, typed-decisions/).
        var onnxRoot = Environment.GetEnvironmentVariable("LAYA_ONNX_ROOT");
        if (!string.IsNullOrEmpty(onnxRoot))
        {
            var rootDir = Path.Combine(onnxRoot, subfolder);
            if (IsComplete(rootDir))
                return ValidateDirectory(rootDir, options.Checkpoint);
            // Not complete under LAYA_ONNX_ROOT — fall through to cache and download rather
            // than failing immediately, so a partially-downloaded cache still works.
        }

        // 4. Cache hit.
        var cacheRoot = ResolveCacheRoot(options);
        var artifactDir = Path.Combine(cacheRoot, subfolder);
        if (IsComplete(artifactDir))
            return ValidateDirectory(artifactDir, options.Checkpoint);

        // 5. Download or report the locations that were tried.
        if (!options.AllowDownload)
        {
            throw new DirectoryNotFoundException(
                $"Laya model artifact (~1.3 GB) not found. Tried:\n" +
                $"  • LayaOptions.ModelDirectory    (not set)\n" +
                $"  • LAYA_ONNX_DIR env var         (not set)\n" +
                $"  • LAYA_ONNX_ROOT env var        {(string.IsNullOrEmpty(onnxRoot) ? "(not set)" : Path.Combine(onnxRoot, subfolder))}\n" +
                $"  • Cache directory               {artifactDir}\n\n" +
                "To fix this, do one of:\n" +
                "  a) Download the artifact and set LAYA_ONNX_DIR or LayaOptions.ModelDirectory " +
                "to its directory, or set LAYA_ONNX_ROOT to the parent of all checkpoints.\n" +
                "  b) Set LayaOptions.AllowDownload = true to download automatically.\n" +
                "Expected one of two layouts:\n" +
                "  Fused : model.onnx + model.onnx.data (both must sit in the same directory —\n" +
                "          ORT resolves the external weights file relative to the graph file).\n" +
                "  Split : encoder.onnx + head.onnx (the laya-ts exporter's layout).");
        }

        await DownloadArtifactAsync(options, artifactDir, ct).ConfigureAwait(false);
        return ValidateDirectory(artifactDir, options.Checkpoint);
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    private static string ResolveCacheRoot(LayaOptions options)
    {
        if (!string.IsNullOrEmpty(options.CacheDirectory))
            return options.CacheDirectory;

        // %LOCALAPPDATA% on Windows; ~/.cache elsewhere (DoNotVerify avoids creation).
        var appData = Environment.GetFolderPath(
            Environment.SpecialFolder.LocalApplicationData,
            Environment.SpecialFolderOption.DoNotVerify);

        if (string.IsNullOrEmpty(appData))
        {
            var home = Environment.GetFolderPath(
                Environment.SpecialFolder.UserProfile,
                Environment.SpecialFolderOption.DoNotVerify);
            appData = Path.Combine(home, ".cache");
        }

        return Path.Combine(appData, "laya", "onnx");
    }

    private static bool IsComplete(string dir)
        => System.IO.Directory.Exists(dir)
        && File.Exists(Path.Combine(dir, "rl_agent_config.json"))
        && ResolveTokenizerPath(dir) is not null
        && (IsFusedComplete(dir) || IsSplitComplete(dir));

    private static bool IsFusedComplete(string dir)
        => File.Exists(Path.Combine(dir, "model.onnx"))
        && File.Exists(Path.Combine(dir, "model.onnx.data"));

    private static bool IsSplitComplete(string dir)
        => File.Exists(Path.Combine(dir, "encoder.onnx"))
        && File.Exists(Path.Combine(dir, "head.onnx"));

    /// <summary>
    /// Looks for <c>tokenizer.json</c> first under <c>&lt;dir&gt;/tokenizer/</c> (the fused
    /// layout's location), then directly in <paramref name="dir"/> (where the split layout's
    /// exporter puts it, alongside <c>rl_agent_config.json</c>). Returns <see langword="null"/>
    /// if found in neither place.
    /// </summary>
    private static string? ResolveTokenizerPath(string dir)
    {
        var nested = Path.Combine(dir, "tokenizer", "tokenizer.json");
        if (File.Exists(nested)) return nested;

        var root = Path.Combine(dir, "tokenizer.json");
        return File.Exists(root) ? root : null;
    }

    private static ModelArtifacts ValidateDirectory(string dir, LayaCheckpoint? checkpoint)
    {
        var requested = dir;
        dir = Path.GetFullPath(dir);

        // Reported before the per-file checks, and quoting the requested path beside the resolved one,
        // because the two diverge in ways that are invisible otherwise: a POSIX shell eats unquoted
        // backslashes, and Windows then reads what is left of "D:\a\b" as the *drive-relative* "D:ab",
        // resolving it against the current directory. Naming only the resolved path turns that into a
        // baffling "missing model.onnx" in a directory the user never typed.
        if (!System.IO.Directory.Exists(dir))
        {
            var note = requested == dir
                ? string.Empty
                : $"\nRequested : {requested}";
            throw new DirectoryNotFoundException(
                $"Laya artifact directory does not exist.{note}\nResolved  : {dir}\n\n" +
                "If the resolved path does not look like what you typed, quote it: an unquoted " +
                "Windows path loses its backslashes in a POSIX shell. Forward slashes also work.");
        }

        // Layout detection: fused wins when both are present, since it is the layout this SDK
        // has always produced and the one most artifact directories will contain.
        var modelPath = Path.Combine(dir, "model.onnx");
        var encoderPath = Path.Combine(dir, "encoder.onnx");
        var headPath = Path.Combine(dir, "head.onnx");

        if (File.Exists(modelPath))
        {
            RequireFile(dir, "model.onnx.data",
                "The external weights file must sit next to model.onnx — ONNX Runtime resolves " +
                "it relative to the graph file, and loading the graph on its own produces a " +
                "baffling 'missing initializer' error.");
            RequireFile(dir, "rl_agent_config.json");
            var tokenizerPath = ResolveTokenizerPath(dir) ?? throw MissingTokenizer(dir);

            return new ModelArtifacts(dir, LayaArtifactLayout.Fused,
                modelPath: modelPath, encoderPath: null, headPath: null,
                configPath: Path.Combine(dir, "rl_agent_config.json"),
                tokenizerPath: tokenizerPath, checkpoint: checkpoint);
        }

        if (File.Exists(encoderPath) && File.Exists(headPath))
        {
            RequireFile(dir, "rl_agent_config.json");
            var tokenizerPath = ResolveTokenizerPath(dir) ?? throw MissingTokenizer(dir);

            return new ModelArtifacts(dir, LayaArtifactLayout.Split,
                modelPath: null, encoderPath: encoderPath, headPath: headPath,
                configPath: Path.Combine(dir, "rl_agent_config.json"),
                tokenizerPath: tokenizerPath, checkpoint: checkpoint);
        }

        // Neither layout is present: half a split export (only one of encoder.onnx/head.onnx)
        // falls through to here too, since neither "if" above matched.
        throw new FileNotFoundException(
            $"No ONNX artifact found in '{dir}'. Expected one of:\n" +
            "  Fused : model.onnx (+ model.onnx.data) next to rl_agent_config.json and\n" +
            "          tokenizer/tokenizer.json.\n" +
            "  Split : encoder.onnx + head.onnx (the laya-ts exporter's layout) next to\n" +
            "          rl_agent_config.json and tokenizer.json.\n",
            modelPath);
    }

    private static FileNotFoundException MissingTokenizer(string dir)
    {
        var expected = Path.Combine(dir, "tokenizer", "tokenizer.json");
        return new FileNotFoundException(
            $"Missing artifact file: {expected}\n" +
            $"Looked for it at '{expected}' and at '{Path.Combine(dir, "tokenizer.json")}'.",
            expected);
    }

    private static void RequireFile(string dir, string relative, string? extraHint = null)
    {
        var full = Path.Combine(dir, relative);
        if (!File.Exists(full))
        {
            var msg = $"Missing artifact file: {full}";
            if (extraHint is not null) msg += "\n" + extraHint;
            throw new FileNotFoundException(msg, full);
        }
    }

    // ── download ───────────────────────────────────────────────────────────────

    // Files to fetch for the fused layout, as (remote sub-path relative to subfolder, local relative path).
    private static readonly (string Remote, string Local)[] FusedArtifactFiles =
    [
        ("model.onnx",                       "model.onnx"),
        ("model.onnx.data",                  "model.onnx.data"),
        ("rl_agent_config.json",             "rl_agent_config.json"),
        ("tokenizer/tokenizer.json",         Path.Combine("tokenizer", "tokenizer.json")),
        ("tokenizer/tokenizer_config.json",  Path.Combine("tokenizer", "tokenizer_config.json")),
    ];

    // Files to fetch for the split layout. No tokenizer_config.json: the exporter does not
    // produce one (see HfTokenizer's added_tokens fallback).
    private static readonly (string Remote, string Local)[] SplitArtifactFiles =
    [
        ("encoder.onnx",           "encoder.onnx"),
        ("encoder.onnx.data",      "encoder.onnx.data"),
        ("head.onnx",              "head.onnx"),
        ("head.onnx.data",         "head.onnx.data"),
        ("rl_agent_config.json",   "rl_agent_config.json"),
        ("tokenizer.json",         "tokenizer.json"),
    ];

    // External-data sidecars are only written when a graph's weights exceed the protobuf limit,
    // so a split export may ship without them; these are skipped on 404 instead of failing.
    private static readonly HashSet<string> OptionalSplitFiles =
        ["encoder.onnx.data", "head.onnx.data"];

    private static async Task DownloadArtifactAsync(
        LayaOptions options, string destDir, CancellationToken ct)
    {
        System.IO.Directory.CreateDirectory(destDir);

        var token = options.HuggingFaceToken
            ?? Environment.GetEnvironmentVariable("HF_TOKEN");

        // One client for every file; disposed here, not per file.
        using var client = new HttpClient();
        if (!string.IsNullOrEmpty(token))
            client.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);

        // The English checkpoint lives at the bundle root; all others use a named subfolder.
        // HuggingFaceDownloadSubfolder returns "" for English, so the URL has no extra segment.
        var hfSub = options.HuggingFaceDownloadSubfolder;
        var baseUrl = $"https://huggingface.co/{options.HuggingFaceRepo}/resolve/main/"
            + (string.IsNullOrEmpty(hfSub) ? "" : hfSub + "/");

        // Try the fused file list first (the layout this SDK has always produced); fall back to
        // the split list only if the repository has no model.onnx. A HEAD request on model.onnx
        // decides which list to fetch, without downloading anything from the losing layout.
        var (layout, files) = await ProbeLayoutAsync(client, baseUrl, ct).ConfigureAwait(false);
        System.Diagnostics.Trace.TraceInformation(
            "Laya: found '{0}' layout artifacts at {1}", layout, baseUrl);

        if (layout == LayaArtifactLayout.Fused)
            System.IO.Directory.CreateDirectory(Path.Combine(destDir, "tokenizer"));

        foreach (var (remote, local) in files)
        {
            var url = $"{baseUrl}{remote}";
            var dest = Path.Combine(destDir, local);
            var part = dest + ".part";

            if (layout == LayaArtifactLayout.Split && OptionalSplitFiles.Contains(remote)
                && !await ExistsAsync(client, url, ct).ConfigureAwait(false))
                continue;

            await DownloadFileAsync(client, url, remote, dest, part, options.DownloadProgress, ct)
                .ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Decides, with a single HEAD request per candidate, whether the repository publishes the
    /// fused or the split layout, and returns the matching file list. Fused is tried first.
    /// </summary>
    private static async Task<(LayaArtifactLayout Layout, (string Remote, string Local)[] Files)> ProbeLayoutAsync(
        HttpClient client, string baseUrl, CancellationToken ct)
    {
        if (await ExistsAsync(client, baseUrl + "model.onnx", ct).ConfigureAwait(false))
            return (LayaArtifactLayout.Fused, FusedArtifactFiles);

        if (await ExistsAsync(client, baseUrl + "encoder.onnx", ct).ConfigureAwait(false))
            return (LayaArtifactLayout.Split, SplitArtifactFiles);

        throw new DirectoryNotFoundException(
            $"Hugging Face has neither 'model.onnx' nor 'encoder.onnx' at {baseUrl} (404).\n\n" +
            "The Laya repository publishes PyTorch weights (model.safetensors), which this SDK " +
            "cannot load: it needs either a fused ONNX export (model.onnx + model.onnx.data) or " +
            "a split export (encoder.onnx + head.onnx). Unless one of those has been uploaded to " +
            "the repository, --download/AllowDownload cannot succeed.\n\n" +
            "Export the model once with the Python package or laya-ts's exporter, then point the " +
            "SDK at the output directory via LAYA_ONNX_DIR or LayaOptions.ModelDirectory.");
    }

    private static async Task<bool> ExistsAsync(HttpClient client, string url, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Head, url);
        using var response = await client.SendAsync(request, ct).ConfigureAwait(false);
        // Anything other than 404 is treated as "exists" — a 401/403 means the repo is gated but
        // the file is really there, and DownloadFileAsync gives the token-specific error for that.
        return response.StatusCode is not System.Net.HttpStatusCode.NotFound;
    }

    private static async Task DownloadFileAsync(
        HttpClient client,
        string url,
        string displayName,
        string destPath,
        string partPath,
        IProgress<ArtifactDownloadProgress>? progress,
        CancellationToken ct)
    {
        using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct)
            .ConfigureAwait(false);

        // A bare EnsureSuccessStatusCode here surfaces as "404 (Not Found)" with no hint as to which
        // file was missing or what to do about it. The two failures worth naming:
        //   404 — the repository does not publish the ONNX export at all (it ships safetensors, which
        //         this SDK cannot load), or the subfolder is wrong.
        //   401/403 — a gated or private repository that needs a token.
        if (response.StatusCode is System.Net.HttpStatusCode.NotFound)
        {
            throw new DirectoryNotFoundException(
                $"Hugging Face has no '{displayName}' at {url} (404).\n\n" +
                "The Laya repository publishes PyTorch weights (model.safetensors), which this SDK " +
                "cannot load: it needs an ONNX export (either model.onnx + model.onnx.data, or " +
                "encoder.onnx + head.onnx). Unless that export has been uploaded to the repository, " +
                "--download/AllowDownload cannot succeed.\n\n" +
                "Export the model once with the Python package, then point the SDK at the output " +
                "directory via LAYA_ONNX_DIR or LayaOptions.ModelDirectory.");
        }

        if (response.StatusCode is System.Net.HttpStatusCode.Unauthorized
                                or System.Net.HttpStatusCode.Forbidden)
        {
            throw new DirectoryNotFoundException(
                $"Hugging Face refused access to {url} ({(int)response.StatusCode}).\n" +
                "The repository is gated or private. Set LayaOptions.HuggingFaceToken, or the " +
                "HF_TOKEN environment variable, to a token that can read it.");
        }

        response.EnsureSuccessStatusCode();

        var total = response.Content.Headers.ContentLength;

        using var src = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using (var dst = new FileStream(partPath, FileMode.Create, FileAccess.Write,
                                        FileShare.None, 81_920, useAsync: true))
        {
            var buffer = new byte[81_920];
            long received = 0;
            int read;
            while ((read = await src.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
            {
                await dst.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                received += read;
                progress?.Report(new ArtifactDownloadProgress(displayName, received, total));
            }
        }

        // Atomic promotion: a .part file that survived means a previous run was interrupted,
        // not that the destination is valid.
        if (File.Exists(destPath)) File.Delete(destPath);
        File.Move(partPath, destPath);
    }
}
