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

    /// <summary>
    /// The commit this artifact was downloaded from, or <see langword="null"/>. For a download (and
    /// a later hit on that cache folder) it is the commit the Hub reported in the <c>X-Repo-Commit</c>
    /// header of every file, kept in a <c>.laya_revision</c> stamp beside the files, as Python reports the
    /// snapshot SHA. When the header is missing or the files disagree there is no stamp, and it falls
    /// back to the pinned value (<see cref="LayaOptions.Revision"/> or <c>LAYA_REVISION</c>), else
    /// <see langword="null"/>. A local directory that is not a download cache is always
    /// <see langword="null"/>.
    /// </summary>
    public string? Revision { get; private set; }

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
        return await ResolveDirectoryAsync(options, ct).ConfigureAwait(false);
    }

    // Integrity is verified before any file in the checkpoint is parsed or executed, and before the
    // directory's own completeness checks, as Python's verify_digests precedes the config read: a
    // directory that is both incomplete and mismatched reports the digest error first. It runs for a
    // local directory too (onnx_agent.py).
    private static void VerifyDigests(
        string dir, LayaOptions options, string? modelPath, bool split) =>
        LayaRevisions.Verify(
            dir, options.ExpectedSha256, modelPath,
            split
                ? "laya: the 'onnx' / 'onnx_path' digest key names the fused model.onnx, but this checkpoint "
                  + "is the split layout (encoder.onnx + head.onnx); list encoder.onnx and head.onnx by "
                  + "relative path instead"
                : null);

    private static async Task<ModelArtifacts> ResolveDirectoryAsync(
        LayaOptions options, CancellationToken ct)
    {

        // 1. Explicit directory. No checkpoint inference: the caller knows what they loaded.
        if (!string.IsNullOrEmpty(options.ModelDirectory))
            return ValidateDirectory(options.ModelDirectory, checkpoint: null, options);

        // 2. LAYA_ONNX_DIR: backward-compat env var that points directly at the artifact dir.
        //    Treated as an explicit path for the same reason: it may point at any checkpoint.
        var envDir = Environment.GetEnvironmentVariable("LAYA_ONNX_DIR");
        if (!string.IsNullOrEmpty(envDir))
            return ValidateDirectory(envDir, checkpoint: null, options);

        var subfolder = LayaOptions.CheckpointSubfolder(options.Checkpoint);

        // 3. LAYA_ONNX_ROOT: new env var for a tree that holds every checkpoint in named
        //    subdirectories (english/, multilingual/, typed-decisions/).
        var onnxRoot = Environment.GetEnvironmentVariable("LAYA_ONNX_ROOT");
        if (!string.IsNullOrEmpty(onnxRoot))
        {
            var rootDir = Path.Combine(onnxRoot, subfolder);
            if (IsComplete(rootDir))
                return ValidateDirectory(rootDir, options.Checkpoint, options);
            // Not complete under LAYA_ONNX_ROOT — fall through to cache and download rather
            // than failing immediately, so a partially-downloaded cache still works.
        }

        // 4. Cache hit. Python resolves the revision once it leaves local paths, before the Hub is
        //    asked; a pin is cached under <cache>/<revision>/<checkpoint>, so it can never be served
        //    from another revision's files.
        var revision = LayaRevisions.ResolveRevision(options.HuggingFaceRepo, options.Revision);
        var cacheRoot = ResolveCacheRoot(options);
        if (revision is not null)
            cacheRoot = Path.Combine(cacheRoot, LayaRevisions.RevisionFolder(revision));
        var artifactDir = Path.Combine(cacheRoot, subfolder);
        if (IsComplete(artifactDir))
            return ValidateDirectory(artifactDir, options.Checkpoint, options)
                .WithRevision(ReadStamp(artifactDir) ?? revision);

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

        var commit = await DownloadArtifactAsync(options, artifactDir, revision, ct).ConfigureAwait(false);
        return ValidateDirectory(artifactDir, options.Checkpoint, options).WithRevision(commit ?? revision);
    }

    private const string StampName = ".laya_revision";

    private static string? ReadStamp(string dir)
    {
        var path = Path.Combine(dir, StampName);
        if (!File.Exists(path)) return null;
        var text = File.ReadAllText(path).Trim();
        return text.Length == 0 ? null : text;
    }

    private ModelArtifacts WithRevision(string? revision)
    {
        Revision = revision;
        return this;
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

    private static ModelArtifacts ValidateDirectory(string dir, LayaCheckpoint? checkpoint, LayaOptions options)
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
            VerifyDigests(dir, options, modelPath, split: false);
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
            VerifyDigests(dir, options, modelPath: null, split: true);
            RequireFile(dir, "rl_agent_config.json");
            var tokenizerPath = ResolveTokenizerPath(dir) ?? throw MissingTokenizer(dir);

            return new ModelArtifacts(dir, LayaArtifactLayout.Split,
                modelPath: null, encoderPath: encoderPath, headPath: headPath,
                configPath: Path.Combine(dir, "rl_agent_config.json"),
                tokenizerPath: tokenizerPath, checkpoint: checkpoint);
        }

        VerifyDigests(dir, options, modelPath: null, split: false);

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

    // Returns the commit every downloaded file reported in X-Repo-Commit (and stamps it into destDir),
    // or null when a response lacked the header or the files disagreed.
    private static async Task<string?> DownloadArtifactAsync(
        LayaOptions options, string destDir, string? revision, CancellationToken ct)
    {
        System.IO.Directory.CreateDirectory(destDir);
        var stamp = Path.Combine(destDir, StampName);
        if (File.Exists(stamp)) File.Delete(stamp);   // a stale stamp must not outlive a new download
        string? commit = null;
        var consistent = true;

        var token = options.HuggingFaceToken
            ?? Environment.GetEnvironmentVariable("HF_TOKEN");

        // One client for every file; disposed here, not per file.
        // Redirects are followed by hand (HubClient): the Hub names the commit in X-Repo-Commit on the
        // first hop only, and the token must not follow a redirect to another host.
        using var http = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false });
        var client = new HubClient(http, string.IsNullOrEmpty(token) ? null : token);

        // The English checkpoint lives at the bundle root; all others use a named subfolder.
        // HuggingFaceDownloadSubfolder returns "" for English, so the URL has no extra segment.
        var hfSub = options.HuggingFaceDownloadSubfolder;
        // resolve/main/ unless pinned; a pin is percent-encoded as one segment, as huggingface_hub does.
        var baseUrl = $"{options.HubEndpoint.TrimEnd('/')}/{options.HuggingFaceRepo}/resolve/"
            + (revision is null ? "main" : Uri.EscapeDataString(revision)) + "/"
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

            var reported = await DownloadFileAsync(client, url, remote, dest, part, options.DownloadProgress, ct)
                .ConfigureAwait(false);
            if (string.IsNullOrWhiteSpace(reported)) consistent = false;
            else if (commit is null) commit = reported.Trim();
            else if (commit != reported.Trim()) consistent = false;
        }

        if (!consistent || commit is null) return null;
        File.WriteAllText(stamp, commit);
        return commit;
    }

    /// <summary>
    /// Decides, with a single HEAD request per candidate, whether the repository publishes the
    /// fused or the split layout, and returns the matching file list. Fused is tried first.
    /// </summary>
    private static async Task<(LayaArtifactLayout Layout, (string Remote, string Local)[] Files)> ProbeLayoutAsync(
        HubClient client, string baseUrl, CancellationToken ct)
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

    private static async Task<bool> ExistsAsync(HubClient client, string url, CancellationToken ct)
    {
        using var hop = await client.SendAsync(HttpMethod.Head, url, ct).ConfigureAwait(false);
        var response = hop.Response;
        // Anything other than 404 is treated as "exists" — a 401/403 means the repo is gated but
        // the file is really there, and DownloadFileAsync gives the token-specific error for that.
        return response.StatusCode is not System.Net.HttpStatusCode.NotFound;
    }

    /// <summary>An HTTP client that follows redirects itself, noting the first X-Repo-Commit it sees.</summary>
    private sealed class HubClient(HttpClient http, string? token)
    {
        private const int MaxRedirects = 10;

        internal sealed class Hop(HttpResponseMessage response, string? commit) : IDisposable
        {
            public HttpResponseMessage Response { get; } = response;
            public string? Commit { get; } = commit;
            public void Dispose() => Response.Dispose();
        }

        // The Authorization header goes only to the origin of the first URL, never to a host a redirect
        // leads to (the CDN), as huggingface_hub does.
        internal async Task<Hop> SendAsync(HttpMethod method, string url, CancellationToken ct)
        {
            var origin = new Uri(url);
            var current = origin;
            string? commit = null;
            for (var redirects = 0; ; redirects++)
            {
                using var request = new HttpRequestMessage(method, current);
                if (token is not null && current.Scheme == origin.Scheme && current.Authority == origin.Authority)
                    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct)
                    .ConfigureAwait(false);
                if (commit is null && response.Headers.TryGetValues("X-Repo-Commit", out var values))
                    commit = values.FirstOrDefault();

                var status = (int)response.StatusCode;
                if (status is not (301 or 302 or 303 or 307 or 308) || response.Headers.Location is not { } location
                    || redirects >= MaxRedirects)
                    return new Hop(response, commit);
                response.Dispose();
                current = new Uri(current, location);   // a relative Location resolves against this hop
            }
        }
    }

    private static async Task<string?> DownloadFileAsync(
        HubClient client,
        string url,
        string displayName,
        string destPath,
        string partPath,
        IProgress<ArtifactDownloadProgress>? progress,
        CancellationToken ct)
    {
        using var hop = await client.SendAsync(HttpMethod.Get, url, ct).ConfigureAwait(false);
        var response = hop.Response;

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

        var commitHeader = hop.Commit;
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
        return commitHeader;
    }
}
