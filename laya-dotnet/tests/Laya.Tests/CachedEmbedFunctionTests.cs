using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Tier 1: <see cref="CachedEmbedFunction"/> against <c>golden/routing/cached_embed_probe.json</c>, which Python's
/// own <c>cached_embed_fn</c> produced over <c>hashing_embedder.embed / 3.0</c>: per step the texts the embedder
/// saw, the float32 rows (or the error) and <c>cache_info()</c>.
/// </summary>
public sealed class CachedEmbedFunctionTests
{
    private static JsonDocument Probe() => RoutingGoldenData.Load("cached_embed_probe.json");

    public static TheoryData<string> ScenarioNames()
    {
        using var doc = Probe();
        return new TheoryData<string>(
            doc.RootElement.GetProperty("scenarios").EnumerateArray().Select(e => e.GetProperty("name").GetString()!));
    }

    /// <summary><c>_SwitchableEmbedder</c> of dump_routing_golden.py.</summary>
    private sealed class Switchable
    {
        public List<string[]> Calls { get; } = [];
        public string Mode { get; set; } = "normal";
        public HashSet<string> Bad { get; set; } = [];

        public double[][] Embed(IReadOnlyList<string> texts)
        {
            Calls.Add([.. texts]);
            if (Mode == "raise") throw new InvalidOperationException("boom");
            var vecs = HashingEmbedder.Embed(texts).Select(row => row.Select(x => x / 3.0).ToArray()).ToArray();
            if (Mode == "short") return [.. vecs.Take(vecs.Length - 1)];
            if (Mode == "dim32") vecs = [.. vecs.Select(row => row[..32])];
            for (var i = 0; i < texts.Count; i++)
            {
                if (!Bad.Contains(texts[i])) continue;
                vecs[i][0] = double.NaN;
                vecs[i][1] = double.PositiveInfinity;
                vecs[i][2] = double.NegativeInfinity;
            }
            return vecs;
        }
    }

    private static void AssertInfo(JsonElement want, EmbedCacheInfo got)
    {
        Assert.Equal(want.GetProperty("size").GetInt32(), got.Size);
        Assert.Equal(want.GetProperty("maxsize").GetInt32(), got.MaxSize);
        Assert.Equal(want.GetProperty("hits").GetInt64(), got.Hits);
        Assert.Equal(want.GetProperty("misses").GetInt64(), got.Misses);
    }

    [Theory]
    [MemberData(nameof(ScenarioNames))]
    public void MatchesPythonsOwnCachedEmbedFn(string name)
    {
        using var doc = Probe();
        var scenario = doc.RootElement.GetProperty("scenarios").EnumerateArray()
            .Single(e => e.GetProperty("name").GetString() == name);
        var embedder = new Switchable();
        var cached = new CachedEmbedFunction(embedder.Embed, scenario.GetProperty("maxsize").GetInt32());

        var index = 0;
        foreach (var step in scenario.GetProperty("steps").EnumerateArray())
        {
            var where = $"{name} step {index++}";
            embedder.Calls.Clear();
            if (step.GetProperty("op").GetString() == "clear")
            {
                cached.CacheClear();
                AssertInfo(step.GetProperty("info"), cached.CacheInfo());
                continue;
            }

            embedder.Mode = step.GetProperty("mode").GetString()!;
            embedder.Bad = [.. step.GetProperty("bad").EnumerateArray().Select(e => e.GetString()!)];
            var texts = step.GetProperty("texts").EnumerateArray()
                .Select(e => e.ValueKind == JsonValueKind.Null ? null! : e.GetString()!).ToArray();

            double[][]? rows = null;
            Exception? error = null;
            try { rows = cached.Embed(texts); }
            catch (Exception e) { error = e; }

            var wantError = step.GetProperty("error");
            if (wantError.ValueKind != JsonValueKind.Null)
            {
                Assert.True(error is not null, where);
                Assert.IsType(wantError.GetProperty("type").GetString() switch
                {
                    "ValueError" => typeof(ArgumentException),
                    "RuntimeError" => typeof(InvalidOperationException),
                    var other => throw new InvalidOperationException(other),
                }, error);
                Assert.Equal(wantError.GetProperty("message").GetString(), error!.Message);
            }
            else
            {
                Assert.True(error is null, $"{where}: {error}");
                var want = step.GetProperty("rows").EnumerateArray()
                    .Select(r => r.EnumerateArray().Select(x => x.GetDouble()).ToArray()).ToArray();
                Assert.Equal(want.Length, rows!.Length);
                for (var i = 0; i < want.Length; i++)
                    Assert.True(want[i].SequenceEqual(rows[i]), $"{where}: row {i}");
            }

            Assert.Equal(
                step.GetProperty("embed_calls").EnumerateArray()
                    .Select(c => c.EnumerateArray().Select(e => e.GetString()!).ToArray()).ToList(),
                embedder.Calls);
            AssertInfo(step.GetProperty("info"), cached.CacheInfo());
        }
    }

    [Fact]
    public void RefusesMaxsizesPythonRefuses()
    {
        using var doc = Probe();
        foreach (var want in doc.RootElement.GetProperty("bad_maxsize").EnumerateArray())
        {
            var e = Assert.Throws<ArgumentException>(
                () => new CachedEmbedFunction(HashingEmbedder.Embed, want.GetProperty("maxsize").GetInt32()));
            Assert.Equal(want.GetProperty("message").GetString(), e.Message);
        }
        Assert.Throws<ArgumentNullException>(() => new CachedEmbedFunction(null!));
        Assert.Equal(4096, CachedEmbedFunction.DefaultMaxSize);
        Assert.Equal(4096, new CachedEmbedFunction(HashingEmbedder.Embed).CacheInfo().MaxSize);
    }

    [Fact]
    public void ConvertsToAnEmbedFunctionForTheShortlist()
    {
        var embedder = new Switchable();
        LayaEmbedFunction fn = new CachedEmbedFunction(embedder.Embed);
        var labels = LayaShortlist.ShortlistChoice("refund my card", ["card", "refund", "loan", "atm"], fn, k: 2);
        LayaShortlist.ShortlistChoice("refund my card", ["card", "refund", "loan", "atm"], fn, k: 2);
        // The second ranking re-embeds nothing: every text is a hit.
        Assert.Single(embedder.Calls);
        Assert.Equal(2, labels.Count);
    }

    [Fact]
    public void ReturnedRowsAreCopiesTheCallerMayChange()
    {
        var cached = new CachedEmbedFunction(HashingEmbedder.Embed);
        var first = cached.Embed(["a"]);
        var before = first[0][5];
        first[0][5] = 12345.0;
        Assert.Equal(before, cached.Embed(["a"])[0][5]);
    }

    [Fact]
    public async Task TheEmbedCallRunsOutsideTheLock()
    {
        // Two threads each miss on their own text and both reach the embedder: with the lock held across
        // the embed call the second could never enter it while the first waits for it.
        using var bothInside = new CountdownEvent(2);
        var cached = new CachedEmbedFunction(texts =>
        {
            bothInside.Signal();
            Assert.True(bothInside.Wait(TimeSpan.FromSeconds(10)), "the second embed call was blocked by the lock");
            return HashingEmbedder.Embed(texts);
        });
        var a = Task.Run(() => cached.Embed(["one"]));
        var b = Task.Run(() => cached.Embed(["two"]));
        await Task.WhenAll(a, b);
        Assert.Equal(2, cached.CacheInfo().Size);
    }

    [Fact]
    public void ManyThreadsKeepTheCountersAndTheBoundConsistent()
    {
        var cached = new CachedEmbedFunction(HashingEmbedder.Embed, 16);
        const int threads = 8, rounds = 300;
        var occurrences = 0L;
        Parallel.For(0, threads, t =>
        {
            var random = new Random(t);
            for (var i = 0; i < rounds; i++)
            {
                var texts = Enumerable.Range(0, 1 + random.Next(4)).Select(_ => $"text-{random.Next(40)}").ToArray();
                Interlocked.Add(ref occurrences, texts.Length);
                var rows = cached.Embed(texts);
                for (var j = 0; j < texts.Length; j++)
                {
                    var want = HashingEmbedder.EmbedOne(texts[j]).Select(x => (double)(float)x).ToArray();
                    Assert.True(want.SequenceEqual(rows[j]), texts[j]);
                }
            }
        });
        var info = cached.CacheInfo();
        Assert.Equal(occurrences, info.Hits + info.Misses);
        Assert.InRange(info.Size, 1, 16);
    }
}
