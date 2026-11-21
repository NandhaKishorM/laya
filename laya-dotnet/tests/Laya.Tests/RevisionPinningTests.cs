using System.Net;
using System.Security.Cryptography;

namespace Laya.Tests;

/// <summary>
/// Pinned downloads (the <c>resolve/&lt;revision&gt;/</c> URL and the per-revision cache), the digest
/// check's place in the load order, and the router's per-checkpoint revisions and digests. A local
/// <see cref="HttpListener"/> stands in for the Hub; there is no network and no model.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class RevisionPinningTests : IDisposable
{
    private const string Repo = "convaiinnovations/laya";
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "laya-revision-tests-" + Guid.NewGuid().ToString("N"));
    private readonly List<HttpListener> _listeners = [];

    public RevisionPinningTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        foreach (var l in _listeners) { try { l.Close(); } catch (ObjectDisposedException) { } }
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // ── a fake Hub ──────────────────────────────────────────────────────────

    private static readonly string[] FusedFiles =
        ["model.onnx", "model.onnx.data", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"];

    /// <summary>
    /// Serves every fused file of every revision and checkpoint; records "METHOD path auth=header".
    /// <paramref name="redirect"/> "relative" or "absolute" answers each request with a 3xx to
    /// <c>/cdn/...</c> on the same host or to a second server, carrying X-Repo-Commit on that
    /// first hop only; the bytes then come without it, as from the real Hub's CDN.
    /// </summary>
    private (string Endpoint, List<string> Requests) StartHub(
        Func<string, string?>? commitFor = null, string? redirect = null)
    {
        var requests = new List<string>();
        var cdn = redirect == "absolute" ? Listen(null, requests) : null;
        var endpoint = Listen(ctx =>
        {
            var url = ctx.Request.RawUrl ?? "";
            if (redirect is null || url.StartsWith("/cdn/")) return false;
            ctx.Response.StatusCode = redirect == "relative" ? 307 : 302;
            ctx.Response.Headers["Location"] = redirect == "relative" ? "/cdn" + url : cdn + "/cdn" + url;
            if (commitFor?.Invoke(url) is { } commit) ctx.Response.Headers["X-Repo-Commit"] = commit;
            return true;
        }, requests, redirect is null ? commitFor : null);
        return (endpoint, requests);
    }

    private string Listen(Func<HttpListenerContext, bool>? intercept, List<string> requests, Func<string, string?>? commitFor = null)
    {
        for (var attempt = 0; ; attempt++)
        {
            var port = Random.Shared.Next(20000, 60000);
            var listener = new HttpListener();
            listener.Prefixes.Add($"http://localhost:{port}/");
            try { listener.Start(); }
            catch (HttpListenerException) when (attempt < 20) { listener.Close(); continue; }
            _listeners.Add(listener);
            _ = Task.Run(async () =>
            {
                while (listener.IsListening)
                {
                    HttpListenerContext ctx;
                    try { ctx = await listener.GetContextAsync(); }
                    catch (Exception) { return; }
                    lock (requests)
                        requests.Add(ctx.Request.HttpMethod + " " + ctx.Request.RawUrl
                            + " auth=" + ctx.Request.Headers["Authorization"] + " port=" + port);
                    if (intercept?.Invoke(ctx) == true) { ctx.Response.Close(); continue; }
                    var body = System.Text.Encoding.UTF8.GetBytes("content of " + ctx.Request.RawUrl);
                    ctx.Response.StatusCode = 200;
                    if (commitFor?.Invoke(ctx.Request.RawUrl ?? "") is { } commit)
                        ctx.Response.Headers["X-Repo-Commit"] = commit;
                    ctx.Response.ContentLength64 = body.Length;
                    if (ctx.Request.HttpMethod != "HEAD") await ctx.Response.OutputStream.WriteAsync(body);
                    ctx.Response.Close();
                }
            });
            return $"http://localhost:{port}";
        }
    }

    private LayaOptions Download(string endpoint, string? revision = null, string? cache = null) => new()
    {
        AllowDownload = true,
        HuggingFaceRepo = Repo,
        CacheDirectory = cache ?? Path.Combine(_dir, "cache"),
        Checkpoint = LayaCheckpoint.Multilingual,
        Revision = revision,
        HubEndpoint = endpoint,
    };

    // ── download and cache ─────────────────────────────────────────────────

    [Fact]
    public void Unpinned_UsesResolveMain_AndTodaysCacheLayout()
    {
        using var env = new EnvScope();
        var (endpoint, requests) = StartHub();

        var artifacts = ModelArtifacts.Resolve(Download(endpoint));

        Assert.Equal(Path.Combine(_dir, "cache", "multilingual"), artifacts.Directory);
        Assert.Null(artifacts.Revision);
        lock (requests)
        {
            Assert.NotEmpty(requests);
            Assert.All(requests, r => Assert.Contains($"/{Repo}/resolve/main/multilingual/", r));
        }
        foreach (var file in FusedFiles)
            Assert.True(File.Exists(Path.Combine(artifacts.Directory, file.Replace('/', Path.DirectorySeparatorChar))), file);
    }

    private const string Snapshot = "fedcba9876543210fedcba9876543210fedcba98";

    [Fact]
    public void RepoCommitHeader_IsReported_OnDownload_AndOnALaterCacheHit()
    {
        using var env = new EnvScope();
        var (endpoint, requests) = StartHub(_ => Snapshot);

        Assert.Equal(Snapshot, ModelArtifacts.Resolve(Download(endpoint)).Revision);
        Assert.True(File.Exists(Path.Combine(_dir, "cache", "multilingual", ".laya_revision")));

        var options = Download(endpoint);
        options.AllowDownload = false;
        int before;
        lock (requests) before = requests.Count;
        Assert.Equal(Snapshot, ModelArtifacts.Resolve(options).Revision);
        lock (requests) Assert.Equal(before, requests.Count);
    }

    [Theory]
    [InlineData("relative")]
    [InlineData("absolute")]
    public void RepoCommit_IsReadFromTheRedirectHop_WhenTheBytesComeWithoutIt(string redirect)
    {
        using var env = new EnvScope();
        var (endpoint, requests) = StartHub(_ => Snapshot, redirect);

        var artifacts = ModelArtifacts.Resolve(Download(endpoint));

        Assert.Equal(Snapshot, artifacts.Revision);
        Assert.True(File.Exists(Path.Combine(_dir, "cache", "multilingual", ".laya_revision")));
        foreach (var file in FusedFiles)
            Assert.StartsWith("content of /cdn/", File.ReadAllText(
                Path.Combine(artifacts.Directory, file.Replace('/', Path.DirectorySeparatorChar))));
        lock (requests) Assert.Contains(requests, r => r.StartsWith("GET /cdn/"));

        var options = Download(endpoint);
        options.AllowDownload = false;
        Assert.Equal(Snapshot, ModelArtifacts.Resolve(options).Revision);
    }

    [Fact]
    public void RedirectWithoutRepoCommit_KeepsTheOldBehaviour()
    {
        using var env = new EnvScope();
        var (endpoint, _) = StartHub(null, "relative");
        Assert.Null(ModelArtifacts.Resolve(Download(endpoint)).Revision);
    }

    [Fact]
    public void Token_IsSentToTheHubHost_AndNotToAnotherHostARedirectLeadsTo()
    {
        using var env = new EnvScope();
        var (endpoint, requests) = StartHub(_ => Snapshot, "absolute");
        var options = Download(endpoint);
        options.HuggingFaceToken = "secret";

        ModelArtifacts.Resolve(options);

        var hubPort = endpoint[(endpoint.LastIndexOf(':') + 1)..];
        lock (requests)
        {
            Assert.All(requests.Where(r => r.EndsWith("port=" + hubPort)), r => Assert.Contains("auth=Bearer secret", r));
            var cdn = requests.Where(r => !r.EndsWith("port=" + hubPort)).ToList();
            Assert.NotEmpty(cdn);
            Assert.All(cdn, r => Assert.DoesNotContain("secret", r));
        }
    }

    [Fact]
    public void RepoCommitHeader_BeatsABranchPin_AndSurvivesInItsFolder()
    {
        using var env = new EnvScope();
        var (endpoint, _) = StartHub(_ => Snapshot);
        Assert.Equal(Snapshot, ModelArtifacts.Resolve(Download(endpoint, "main")).Revision);
        var options = Download(endpoint, "main");
        options.AllowDownload = false;
        Assert.Equal(Snapshot, ModelArtifacts.Resolve(options).Revision);
    }

    [Fact]
    public void NoRepoCommitHeader_FallsBackToThePinnedValue_ElseNull()
    {
        using var env = new EnvScope();
        var (endpoint, _) = StartHub();
        Assert.Null(ModelArtifacts.Resolve(Download(endpoint)).Revision);
        Assert.False(File.Exists(Path.Combine(_dir, "cache", "multilingual", ".laya_revision")));
        var pinned = Download(endpoint, "rev-a");
        Assert.Equal("rev-a", ModelArtifacts.Resolve(pinned).Revision);
        pinned.AllowDownload = false;
        Assert.Equal("rev-a", ModelArtifacts.Resolve(pinned).Revision);
    }

    [Fact]
    public void DisagreeingOrPartialRepoCommit_FallsBack_AndWritesNoStamp()
    {
        using var env = new EnvScope();
        var (endpoint, _) = StartHub(url => url.EndsWith("rl_agent_config.json") ? "1111" : Snapshot);
        Assert.Equal("rev-a", ModelArtifacts.Resolve(Download(endpoint, "rev-a")).Revision);
        Assert.False(File.Exists(Path.Combine(_dir, "cache", "rev-a", "multilingual", ".laya_revision")));

        var (partial, _) = StartHub(url => url.EndsWith("model.onnx.data") ? null : Snapshot);
        Assert.Null(ModelArtifacts.Resolve(Download(partial, cache: Path.Combine(_dir, "cache2"))).Revision);
    }

    [Fact]
    public void LocalDirectory_NeverReportsAStampedCommit()
    {
        using var env = new EnvScope();
        WriteFusedCheckpoint(_dir);
        Assert.Null(ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = _dir }).Revision);
    }

    [Fact]
    public void Pinned_UsesResolveRevision_AndCachesPerRevision()
    {
        using var env = new EnvScope();
        var (endpoint, requests) = StartHub();
        const string sha = "0123456789abcdef0123456789abcdef01234567";

        var artifacts = ModelArtifacts.Resolve(Download(endpoint, sha));

        Assert.Equal(Path.Combine(_dir, "cache", sha, "multilingual"), artifacts.Directory);
        Assert.Equal(sha, artifacts.Revision);
        lock (requests)
        {
            Assert.NotEmpty(requests);
            Assert.All(requests, r => Assert.Contains($"/{Repo}/resolve/{sha}/multilingual/", r));
            Assert.DoesNotContain(requests, r => r.Contains("/resolve/main/"));
        }
    }

    [Fact]
    public void Pinned_IsServedFromItsOwnCacheFolder_AndNeverFromAnotherRevisions()
    {
        using var env = new EnvScope();
        var (endpoint, requests) = StartHub();
        ModelArtifacts.Resolve(Download(endpoint, "rev-a"));
        int afterFirst;
        lock (requests) afterFirst = requests.Count;

        // Same revision again: a cache hit, no request at all (even with downloads off).
        var options = Download(endpoint, "rev-a");
        options.AllowDownload = false;
        Assert.Equal("rev-a", ModelArtifacts.Resolve(options).Revision);
        lock (requests) Assert.Equal(afterFirst, requests.Count);

        // Another revision, and no pin, are not satisfied by it.
        options.Revision = "rev-b";
        Assert.Throws<DirectoryNotFoundException>(() => ModelArtifacts.Resolve(options));
        options.Revision = null;
        Assert.Throws<DirectoryNotFoundException>(() => ModelArtifacts.Resolve(options));
    }

    [Fact]
    public void Pinned_BranchWithSlash_IsOneEncodedSegmentInUrlAndFolder()
    {
        using var env = new EnvScope();
        var (endpoint, requests) = StartHub();

        var artifacts = ModelArtifacts.Resolve(Download(endpoint, "refs/pr/1"));

        Assert.Equal(Path.Combine(_dir, "cache", "refs%2Fpr%2F1", "multilingual"), artifacts.Directory);
        lock (requests) Assert.All(requests, r => Assert.Contains("/resolve/refs%2Fpr%2F1/multilingual/", r));
    }

    [Theory]
    [InlineData("..", "%2E%2E")]
    [InlineData(".", "%2E")]
    [InlineData("v1.", "v1%2E")]
    [InlineData("v0.4.16", "v0.4.16")]
    [InlineData("a/../b", "a%2F..%2Fb")]
    [InlineData("a\\b", "a%5Cb")]
    public void RevisionFolder_IsASafeSingleSegment(string revision, string folder) =>
        Assert.Equal(folder, LayaRevisions.RevisionFolder(revision));

    [Fact]
    public void ReviewedFromEnvironment_DownloadsThePinnedSha()
    {
        using var env = new EnvScope(("LAYA_REVISION", " reviewed "));
        var (endpoint, requests) = StartHub();
        var sha = LayaRevisions.PinnedRevisions[Repo];

        var artifacts = ModelArtifacts.Resolve(Download(endpoint));

        Assert.Equal(sha, artifacts.Revision);
        lock (requests) Assert.All(requests, r => Assert.Contains($"/resolve/{sha}/multilingual/", r));
    }

    [Fact]
    public void EnvironmentRevision_AppliesWhenNoOptionIsSet_AndOptionWins()
    {
        using var env = new EnvScope(("LAYA_REVISION", "from-env"));
        var (endpoint, requests) = StartHub();

        Assert.Equal("from-env", ModelArtifacts.Resolve(Download(endpoint)).Revision);
        Assert.Equal("from-option", ModelArtifacts.Resolve(Download(endpoint, "from-option")).Revision);
        Assert.Null(ModelArtifacts.Resolve(Download(endpoint, "   ")).Revision);
    }

    [Fact]
    public void ReviewedForAnUnknownRepository_IsRefusedBeforeAnyRequest()
    {
        using var env = new EnvScope();
        var (endpoint, requests) = StartHub();
        var options = Download(endpoint, "reviewed");
        options.HuggingFaceRepo = "someone/else";

        var ex = Assert.Throws<ArgumentException>(() => ModelArtifacts.Resolve(options));

        Assert.StartsWith("laya: revision= asked for 'reviewed', but 'someone/else'", ex.Message);
        lock (requests) Assert.Empty(requests);
    }

    [Fact]
    public void LocalDirectory_IgnoresRevision_AndReportsNone()
    {
        using var env = new EnvScope();
        WriteFusedCheckpoint(_dir);

        var artifacts = ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = _dir, Revision = "reviewed" });

        Assert.Null(artifacts.Revision);
    }

    // ── digests: where the check runs ──────────────────────────────────────

    private static void WriteFusedCheckpoint(string dir, string config = "{}")
    {
        Directory.CreateDirectory(Path.Combine(dir, "tokenizer"));
        File.WriteAllText(Path.Combine(dir, "model.onnx"), "not a graph");
        File.WriteAllText(Path.Combine(dir, "model.onnx.data"), "weights");
        File.WriteAllText(Path.Combine(dir, "rl_agent_config.json"), config);
        File.WriteAllText(Path.Combine(dir, "tokenizer", "tokenizer.json"), "{}");
    }

    private static void WriteSplitCheckpoint(string dir)
    {
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "encoder.onnx"), "encoder");
        File.WriteAllText(Path.Combine(dir, "head.onnx"), "head");
        File.WriteAllText(Path.Combine(dir, "rl_agent_config.json"), "{}");
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), "{}");
    }

    private static string Sha(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));

    [Fact]
    public void Digests_AreVerifiedForALocalDirectory()
    {
        using var env = new EnvScope();
        WriteFusedCheckpoint(_dir);
        var good = Sha(Path.Combine(_dir, "model.onnx"));

        var ok = ModelArtifacts.Resolve(new LayaOptions
        {
            ModelDirectory = _dir,
            ExpectedSha256 = new Dictionary<string, string> { ["model.onnx"] = good.ToUpperInvariant() },
        });
        Assert.Equal(LayaArtifactLayout.Fused, ok.Layout);

        var ex = Assert.Throws<ArgumentException>(() => ModelArtifacts.Resolve(new LayaOptions
        {
            ModelDirectory = _dir,
            ExpectedSha256 = new Dictionary<string, string> { ["model.onnx"] = "00" },
        }));
        Assert.StartsWith("laya: SHA-256 mismatch for model.onnx: expected 00, got " + good, ex.Message);
    }

    [Fact]
    public void Digests_AreVerifiedBeforeTheDirectoryIsCheckedForCompleteness()
    {
        using var env = new EnvScope();
        // model.onnx.data, the config and the tokenizer are all missing: a digest mismatch is reported
        // first, then (once the digests agree) the missing file, as Python verifies before reading the config.
        File.WriteAllText(Path.Combine(_dir, "model.onnx"), "not a graph");
        var wrong = new Dictionary<string, string> { ["model.onnx"] = "00" };
        var right = new Dictionary<string, string> { ["model.onnx"] = Sha(Path.Combine(_dir, "model.onnx")) };

        Assert.StartsWith("laya: SHA-256 mismatch for model.onnx", Assert.Throws<ArgumentException>(() =>
            ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = _dir, ExpectedSha256 = wrong })).Message);
        Assert.Contains("model.onnx.data", Assert.Throws<FileNotFoundException>(() =>
            ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = _dir, ExpectedSha256 = right })).Message);

        var split = Path.Combine(_dir, "split");
        Directory.CreateDirectory(split);
        File.WriteAllText(Path.Combine(split, "encoder.onnx"), "e");
        File.WriteAllText(Path.Combine(split, "head.onnx"), "h");
        Assert.StartsWith("laya: SHA-256 mismatch for head.onnx", Assert.Throws<ArgumentException>(() =>
            ModelArtifacts.Resolve(new LayaOptions
            {
                ModelDirectory = split,
                ExpectedSha256 = new Dictionary<string, string> { ["head.onnx"] = "00" },
            })).Message);
    }

    [Fact]
    public void Digests_AreVerifiedBeforeConfigTokenizerOrSessionLoad()
    {
        using var env = new EnvScope();
        // The config is not JSON and the graph is not ONNX: a load would fail on either, but a wrong
        // digest must be reported first, and a right one must get past the check.
        WriteFusedCheckpoint(_dir, config: "this is not json");
        var wrong = new Dictionary<string, string> { ["model.onnx.data"] = "00" };
        var right = new Dictionary<string, string> { ["model.onnx.data"] = Sha(Path.Combine(_dir, "model.onnx.data")) };

        Assert.StartsWith("laya: SHA-256 mismatch for model.onnx.data",
            Assert.Throws<ArgumentException>(() => LayaEngine.Create(
                new LayaOptions { ModelDirectory = _dir, ExpectedSha256 = wrong })).Message);

        var past = Assert.ThrowsAny<Exception>(() => LayaEngine.Create(
            new LayaOptions { ModelDirectory = _dir, ExpectedSha256 = right }));
        Assert.DoesNotContain("SHA-256", past.Message);
    }

    [Fact]
    public void Digests_FallBackToTheEnvironment_AndAnEmptyMapMasksIt()
    {
        WriteFusedCheckpoint(_dir);
        using (new EnvScope(("LAYA_SHA256_DIGESTS", "{\"model.onnx\": \"00\"}")))
        {
            Assert.Throws<ArgumentException>(() => ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = _dir }));
            var masked = ModelArtifacts.Resolve(new LayaOptions
            {
                ModelDirectory = _dir,
                ExpectedSha256 = new Dictionary<string, string>(),
            });
            Assert.Equal(LayaArtifactLayout.Fused, masked.Layout);
        }
    }

    [Fact]
    public void OnnxKey_NamesTheFusedModel_AndIsRefusedOnASplitCheckpoint()
    {
        using var env = new EnvScope();
        var fused = Path.Combine(_dir, "fused");
        var split = Path.Combine(_dir, "split");
        WriteFusedCheckpoint(fused);
        WriteSplitCheckpoint(split);

        foreach (var key in new[] { "onnx", "onnx_path" })
        {
            var onFused = new Dictionary<string, string> { [key] = Sha(Path.Combine(fused, "model.onnx")) };
            Assert.Equal(LayaArtifactLayout.Fused,
                ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = fused, ExpectedSha256 = onFused }).Layout);

            var onSplit = new Dictionary<string, string> { [key] = "00" };
            var ex = Assert.Throws<ArgumentException>(() =>
                ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = split, ExpectedSha256 = onSplit }));
            Assert.Contains("split layout", ex.Message);
            Assert.Contains("encoder.onnx and head.onnx", ex.Message);

            // The split files themselves are listed by path.
            var byPath = new Dictionary<string, string>
            {
                ["encoder.onnx"] = Sha(Path.Combine(split, "encoder.onnx")),
                ["head.onnx"] = Sha(Path.Combine(split, "head.onnx")),
            };
            Assert.Equal(LayaArtifactLayout.Split,
                ModelArtifacts.Resolve(new LayaOptions { ModelDirectory = split, ExpectedSha256 = byPath }).Layout);
        }
    }

    [Fact]
    public void PinnedDownload_IsVerifiedAfterDownload()
    {
        using var env = new EnvScope();
        var (endpoint, _) = StartHub();
        var options = Download(endpoint, "rev-a");
        options.ExpectedSha256 = new Dictionary<string, string> { ["model.onnx"] = "00" };

        Assert.StartsWith("laya: SHA-256 mismatch for model.onnx",
            Assert.Throws<ArgumentException>(() => ModelArtifacts.Resolve(options)).Message);
    }

    // ── router ────────────────────────────────────────────────────────────

    private sealed class Pinned(string? revision) : ILayaPredictor, IRevisioned
    {
        public string? Revision { get; } = revision;
        public LayaResult Predict(object? state, QuestionSet questions) => throw new NotSupportedException();
    }

    private sealed class Unpinned : ILayaPredictor
    {
        public LayaResult Predict(object? state, QuestionSet questions) => throw new NotSupportedException();
    }

    private static LayaRouter NewRouter(LayaRouterOptions? options = null) =>
        new(options, _ => new Unpinned());

    [Fact]
    public void Router_RevisionsOverrideRevision_AndABlankEntryInherits()
    {
        using var env = new EnvScope();
        using var router = NewRouter(new LayaRouterOptions
        {
            Revision = "shared",
            Revisions = new Dictionary<string, string?> { ["en"] = "own", ["ml"] = "   ", ["typed"] = null },
        });

        Assert.Equal("own", router.OptionsFor(LayaCheckpoint.English).Revision);
        Assert.Equal("shared", router.OptionsFor(LayaCheckpoint.Multilingual).Revision);
        Assert.Equal("shared", router.OptionsFor(LayaCheckpoint.TypedDecisions).Revision);
    }

    [Fact]
    public void Router_BlankRevision_PinsNothing_SoTheEnvironmentStaysInCharge()
    {
        using var env = new EnvScope();
        using var router = NewRouter(new LayaRouterOptions { Revision = "  " });
        Assert.Null(router.OptionsFor(LayaCheckpoint.English).Revision);
    }

    [Fact]
    public void Router_UnknownRevisionsName_ThrowsAtConstruction_WithPythonsText()
    {
        using var env = new EnvScope();
        var ex = Assert.Throws<ArgumentException>(() => NewRouter(new LayaRouterOptions
        {
            Revisions = new Dictionary<string, string?> { ["gpt"] = "x" },
        }));
        Assert.StartsWith("unknown model 'gpt'; choose one of ", ex.Message);
    }

    [Fact]
    public void Router_DigestsMergeFileByFile_AndPerCheckpointWins()
    {
        using var env = new EnvScope();
        var template = new LayaOptions
        {
            ExpectedSha256 = new Dictionary<string, string> { ["model.safetensors"] = "shared", ["config"] = "c" },
        };
        using var router = NewRouter(new LayaRouterOptions
        {
            EngineOptions = template,
            Sha256Digests = new Dictionary<string, IReadOnlyDictionary<string, string>?>
            {
                ["english"] = new Dictionary<string, string> { ["model.safetensors"] = "own", ["extra"] = "e" },
            },
        });

        var english = router.OptionsFor(LayaCheckpoint.English).ExpectedSha256!;
        Assert.Equal(("own", "c", "e"), (english["model.safetensors"], english["config"], english["extra"]));
        var other = router.OptionsFor(LayaCheckpoint.Multilingual).ExpectedSha256!;
        Assert.Equal(("shared", "c"), (other["model.safetensors"], other["config"]));
        Assert.Equal(2, other.Count);
    }

    [Fact]
    public void Router_NoDigests_LeavesTheLoadersEnvironmentFallback()
    {
        using var env = new EnvScope();
        using var router = NewRouter();
        Assert.Null(router.OptionsFor(LayaCheckpoint.English).ExpectedSha256);
    }

    [Fact]
    public void Router_FlatEnvironment_IsLeftToTheLoader_NestedPinsEachCheckpoint()
    {
        using (new EnvScope(("LAYA_SHA256_DIGESTS", "{\"model.onnx\": \"aa\"}")))
        {
            using var flat = NewRouter();
            Assert.Null(flat.OptionsFor(LayaCheckpoint.English).ExpectedSha256);
        }
        using (new EnvScope(("LAYA_SHA256_DIGESTS", "{\"en\": {\"model.onnx\": \"aa\"}}")))
        {
            using var nested = NewRouter();
            Assert.Equal("aa", nested.OptionsFor(LayaCheckpoint.English).ExpectedSha256!["model.onnx"]);
            // Named by nothing: an empty map that masks the nested variable for this checkpoint.
            Assert.Empty(nested.OptionsFor(LayaCheckpoint.Multilingual).ExpectedSha256!);
        }
    }

    [Fact]
    public void Router_ArgumentEntryReplacesTheEnvironmentsForThatCheckpoint_AndNullMasksIt()
    {
        using var env = new EnvScope(("LAYA_SHA256_DIGESTS",
            "{\"en\": {\"a\": \"1\"}, \"ml\": {\"b\": \"2\"}, \"typed\": {\"c\": \"3\"}}"));
        using var router = NewRouter(new LayaRouterOptions
        {
            Sha256Digests = new Dictionary<string, IReadOnlyDictionary<string, string>?>
            {
                ["english"] = new Dictionary<string, string> { ["x"] = "9" },
                ["multilingual"] = null,
            },
        });

        var english = router.OptionsFor(LayaCheckpoint.English).ExpectedSha256!;
        Assert.Equal([("x", "9")], english.Select(kv => (kv.Key, kv.Value)).ToArray());
        Assert.Empty(router.OptionsFor(LayaCheckpoint.Multilingual).ExpectedSha256!);
        Assert.Equal("3", router.OptionsFor(LayaCheckpoint.TypedDecisions).ExpectedSha256!["c"]);
    }

    [Fact]
    public void Router_MixedEnvironmentShape_ThrowsAtConstruction_WithPythonsText()
    {
        using var env = new EnvScope(("LAYA_SHA256_DIGESTS", "{\"model.onnx\": \"aa\", \"english\": {}}"));
        var ex = Assert.Throws<ArgumentException>(() => NewRouter());
        Assert.Equal(
            "LAYA_SHA256_DIGESTS must be either {artifact: digest} for every checkpoint or "
            + "{model: {artifact: digest}} per checkpoint; ['english', 'model.onnx'] mixes the two or holds "
            + "a value that is neither", ex.Message);
    }

    [Fact]
    public void Router_TemplateOptionsAreCarried_ButNotItsRevision()
    {
        using var env = new EnvScope();
        using var router = NewRouter(new LayaRouterOptions
        {
            EngineOptions = new LayaOptions { Revision = "ignored", HuggingFaceRepo = "me/laya", AllowDownload = true },
        });
        var options = router.OptionsFor(LayaCheckpoint.English);
        Assert.Null(options.Revision);
        Assert.Equal("me/laya", options.HuggingFaceRepo);
        Assert.True(options.AllowDownload);
    }

    [Fact]
    public void LoadedRevisions_ReportsEachResidentCheckpoint_NullWhenUnknown()
    {
        using var env = new EnvScope();
        using var router = new LayaRouter(new LayaRouterOptions { MaxLoaded = 3 },
            ck => ck == LayaCheckpoint.English ? new Pinned("abc") : new Unpinned());

        Assert.Empty(router.LoadedRevisions);
        router.Load("english");
        router.Load("multilingual");
        router.Attach("typed", new Pinned(null));

        var loaded = router.LoadedRevisions;
        Assert.Equal(3, loaded.Count);
        Assert.Equal("abc", loaded[LayaCheckpoint.English]);
        Assert.Null(loaded[LayaCheckpoint.Multilingual]);
        Assert.Null(loaded[LayaCheckpoint.TypedDecisions]);

        router.Unload("english");
        Assert.False(router.LoadedRevisions.ContainsKey(LayaCheckpoint.English));
    }
}
