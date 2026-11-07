using System.Globalization;

namespace Laya;

/// <summary>A snapshot of a <see cref="CachedEmbedFunction"/>: Python's <c>cache_info()</c> dict.</summary>
/// <param name="Size">How many strings the cache holds now.</param>
/// <param name="MaxSize">The bound on <paramref name="Size"/>.</param>
/// <param name="Hits">Texts served from the cache, counted per occurrence in a call.</param>
/// <param name="Misses">Texts that were not in the cache, counted per occurrence in a call.</param>
public readonly record struct EmbedCacheInfo(int Size, int MaxSize, long Hits, long Misses);

/// <summary>
/// Caches a <see cref="LayaEmbedFunction"/>'s output per input string, under an LRU bound. Ports Python's
/// <c>cached_embed_fn</c> (<c>shortlist.py:287-400</c>).
/// </summary>
/// <remarks>
/// <para>
/// <see cref="LayaShortlist.Predict"/> embeds the query plus every option text on each call. When the same
/// option set is shortlisted on every request, the option rows do not change between calls, yet they are
/// re-embedded every time. Wrapping the embedder once leaves the first call's embedder work unchanged and
/// reduces each repeat call to embedding the new query alone.
/// </para>
/// <para>
/// Lookups are exact string matches (a null text is the empty string). Texts missing from the cache are
/// de-duplicated and embedded in a single call, so a cold cache costs the same number of batched calls as the
/// unwrapped function. Rows are stored as 32-bit floats; the cache holds at most <c>maxSize</c> strings and then
/// evicts the least recently used entry. Nothing is cached when the embedder throws or returns a bad shape. A
/// non-finite value is not a bad shape: NaN and infinities become <c>0.0</c> on the way in and the repaired row
/// is then <em>cached</em>, so one transient NaN ranks that option as a zero vector on every later request until
/// <see cref="CacheClear"/>. The float32 storage shows in the scores too: a wrapped first call can differ from the
/// unwrapped one in its last digits.
/// </para>
/// <para>
/// Safe to share between threads: the lock covers only cache reads and writes, never the embedding call.
/// Clear the cache if the model behind the embedder changes.
/// </para>
/// </remarks>
public sealed class CachedEmbedFunction
{
    private readonly LayaEmbedFunction _embedFn;
    private readonly int _maxSize;
    private readonly object _lock = new();
    private readonly Dictionary<string, LinkedListNode<KeyValuePair<string, float[]>>> _index = new(StringComparer.Ordinal);
    private readonly LinkedList<KeyValuePair<string, float[]>> _rows = new();   // first = least recently used
    private long _hits;
    private long _misses;

    /// <summary>Default bound, matching Python's <c>cached_embed_fn</c>.</summary>
    public const int DefaultMaxSize = 4096;

    /// <summary>Wrap <paramref name="embedFn"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="maxSize"/> is below 1, with Python's <c>ValueError</c> text.</exception>
    public CachedEmbedFunction(LayaEmbedFunction embedFn, int maxSize = DefaultMaxSize)
    {
        ArgumentNullException.ThrowIfNull(embedFn);
        if (maxSize < 1)
            throw new ArgumentException(
                $"maxsize must be a positive integer, got {maxSize.ToString(CultureInfo.InvariantCulture)}");
        _embedFn = embedFn;
        _maxSize = maxSize;
    }

    /// <summary>The cached function, for <see cref="LayaShortlist.Predict"/>.</summary>
    public static implicit operator LayaEmbedFunction(CachedEmbedFunction cached) => cached.Embed;

    /// <summary>
    /// One row per text, in order (float32 values widened to double). An empty list returns an empty array
    /// without touching the embedder.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The embedder returned a bad shape, or a dimension that differs from the one the cache holds, with
    /// Python's <c>ValueError</c> texts.
    /// </exception>
    public double[][] Embed(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var keys = new string[texts.Count];
        for (var i = 0; i < keys.Length; i++) keys[i] = texts[i] ?? "";
        if (keys.Length == 0) return [];

        var found = new Dictionary<string, float[]>(StringComparer.Ordinal);
        List<string> missing;
        lock (_lock)
        {
            var hits = 0;
            foreach (var key in keys)
            {
                if (!_index.TryGetValue(key, out var node)) continue;
                _rows.Remove(node);
                _rows.AddLast(node);
                found[key] = node.Value.Value;
                hits++;
            }
            _hits += hits;
            _misses += keys.Length - hits;
            missing = [];
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var key in keys)
                if (!found.ContainsKey(key) && seen.Add(key)) missing.Add(key);
        }

        if (missing.Count > 0)
        {
            var fresh = Checked(_embedFn(missing), missing.Count);
            lock (_lock)
            {
                if (_rows.First is { } first)
                {
                    // Rows of one dimensionality cannot be stacked against rows of another.
                    var want = first.Value.Value.Length;
                    if (fresh[0].Length != want)
                        throw new ArgumentException(
                            $"embed_fn returned dim {fresh[0].Length}, but the cache holds dim {want}; "
                            + "call cache_clear() if the model behind embed_fn changed");
                }
                for (var i = 0; i < missing.Count; i++)
                {
                    var key = missing[i];
                    // A concurrent call may have inserted the key meanwhile; Python's store replaces and re-orders it.
                    if (_index.Remove(key, out var stale)) _rows.Remove(stale);
                    var node = new LinkedListNode<KeyValuePair<string, float[]>>(new(key, fresh[i]));
                    _index[key] = node;
                    _rows.AddLast(node);
                    while (_index.Count > _maxSize)
                    {
                        var oldest = _rows.First!;
                        _rows.RemoveFirst();
                        _index.Remove(oldest.Value.Key);
                    }
                    found[key] = fresh[i];
                }
            }
        }

        var result = new double[keys.Length][];
        for (var i = 0; i < keys.Length; i++)
        {
            var row = found[keys[i]];
            var copy = new double[row.Length];
            for (var j = 0; j < row.Length; j++) copy[j] = row[j];
            result[i] = copy;
        }
        return result;
    }

    /// <summary>Size, bound and counters, taken under the lock.</summary>
    public EmbedCacheInfo CacheInfo()
    {
        lock (_lock) return new EmbedCacheInfo(_index.Count, _maxSize, _hits, _misses);
    }

    /// <summary>Drop every row and reset the counters.</summary>
    public void CacheClear()
    {
        lock (_lock)
        {
            _index.Clear();
            _rows.Clear();
            _hits = 0;
            _misses = 0;
        }
    }

    // np.asarray(raw, dtype=np.float32) then the (n, dim) check, then np.nan_to_num(nan=0, posinf=0, neginf=0).
    private static float[][] Checked(double[][]? raw, int expected)
    {
        string Bad(string shape) =>
            $"embed_fn must return an array of shape ({expected}, dim), got {shape}";

        if (raw is null) throw new ArgumentException(Bad("()"));
        var rows = raw.Length;
        var nulls = raw.Count(r => r is null);
        if (rows > 0 && nulls == rows) throw new ArgumentException(Bad($"({rows},)"));
        if (rows == 0) throw new ArgumentException(Bad("(0,)"));
        if (nulls > 0 || raw.Any(r => r.Length != raw[0].Length))
            throw new ArgumentException(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 "
                + $"dimensions. The detected shape was ({rows},) + inhomogeneous part.");
        var dim = raw[0].Length;
        if (rows != expected || dim < 1) throw new ArgumentException(Bad($"({rows}, {dim})"));

        var fresh = new float[rows][];
        for (var i = 0; i < rows; i++)
        {
            var row = new float[dim];
            for (var j = 0; j < dim; j++)
            {
                var v = (float)raw[i][j];
                row[j] = float.IsNaN(v) || float.IsInfinity(v) ? 0.0f : v;
            }
            fresh[i] = row;
        }
        return fresh;
    }
}
