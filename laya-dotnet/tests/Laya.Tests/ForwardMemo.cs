namespace Laya.Tests;

/// <summary>
/// Runs byte-identical forward passes once. Test-only: it sits behind
/// <c>LayaEngine.ForwardInterceptor</c> and wraps the graph run only, so every case still
/// tokenizes, collates, calibrates, gates and decodes on its own and is compared with its own golden.
/// </summary>
/// <remarks>
/// The key is the whole collated batch: its three dimensions and the exact contents of every tensor
/// the graph is fed (<c>input_ids</c>, <c>attention_mask</c>, <c>marker_pos</c>, <c>marker_mask</c>,
/// <c>qtype</c>), compared element by element, never through a hash alone. A memo belongs to one
/// checkpoint and one layout (see <see cref="ForwardMemos"/>), which fixes the graph and so the
/// set of input names. A hit returns copies, so a caller can never alter a stored answer, and the
/// result of a lookup does not depend on which case ran first.
/// </remarks>
public sealed class ForwardMemo
{
    private readonly object _gate = new();
    private readonly Dictionary<Key, (float[] Logits, float[] ActLogits)> _stored = [];

    /// <summary>Forward passes that reached the graph.</summary>
    public int Runs { get; private set; }

    /// <summary>Forward passes answered from an identical earlier one.</summary>
    public int Hits { get; private set; }

    /// <summary>Returns the outputs for <paramref name="batch"/>, running <paramref name="run"/> only on a miss.</summary>
    public (float[] Logits, float[] ActLogits) Forward(
        CollatedBatch batch, Func<CollatedBatch, (float[] Logits, float[] ActLogits)> run)
    {
        var key = new Key(batch);
        lock (_gate)
        {
            if (_stored.TryGetValue(key, out var hit))
            {
                Hits++;
                return ((float[])hit.Logits.Clone(), (float[])hit.ActLogits.Clone());
            }
            // Held across the run: the engine's tests are sequential, and this keeps a second
            // caller with the same input from running it again.
            var (logits, actLogits) = run(batch);
            Runs++;
            _stored[key] = ((float[])logits.Clone(), (float[])actLogits.Clone());
            return (logits, actLogits);
        }
    }

    private sealed class Key(CollatedBatch batch) : IEquatable<Key>
    {
        private readonly int[] _shape = [batch.Count, batch.SequenceLength, batch.MarkerCount];
        private readonly long[] _inputIds = (long[])batch.InputIds.Clone();
        private readonly long[] _attentionMask = (long[])batch.AttentionMask.Clone();
        private readonly long[] _markerPos = (long[])batch.MarkerPos.Clone();
        private readonly bool[] _markerMask = (bool[])batch.MarkerMask.Clone();
        private readonly long[] _qtype = (long[])batch.QType.Clone();

        public bool Equals(Key? other) =>
            other is not null
            && _shape.AsSpan().SequenceEqual(other._shape)
            && _inputIds.AsSpan().SequenceEqual(other._inputIds)
            && _attentionMask.AsSpan().SequenceEqual(other._attentionMask)
            && _markerPos.AsSpan().SequenceEqual(other._markerPos)
            && _markerMask.AsSpan().SequenceEqual(other._markerMask)
            && _qtype.AsSpan().SequenceEqual(other._qtype);

        public override bool Equals(object? obj) => Equals(obj as Key);

        // Only a bucket selector; Equals decides identity.
        public override int GetHashCode()
        {
            var hash = new HashCode();
            foreach (var d in _shape) hash.Add(d);
            foreach (var v in _inputIds) hash.Add(v);
            foreach (var v in _markerPos) hash.Add(v);
            foreach (var v in _qtype) hash.Add(v);
            return hash.ToHashCode();
        }
    }
}

/// <summary>One <see cref="ForwardMemo"/> per checkpoint and layout, installed on an engine for one test.</summary>
public static class ForwardMemos
{
    private static readonly Dictionary<(LayaCheckpoint Checkpoint, bool Split), ForwardMemo> Memos = [];

    /// <summary>The memo for one checkpoint and layout, created on first use.</summary>
    public static ForwardMemo For(LayaCheckpoint checkpoint, bool split)
    {
        lock (Memos)
        {
            if (!Memos.TryGetValue((checkpoint, split), out var memo))
                Memos[(checkpoint, split)] = memo = new ForwardMemo();
            return memo;
        }
    }

    /// <summary>
    /// Routes <paramref name="engine"/>'s forward passes through the memo of its checkpoint and
    /// layout until the returned scope is disposed.
    /// </summary>
    public static IDisposable Install(LayaEngine engine, LayaCheckpoint checkpoint, bool split)
    {
        var memo = For(checkpoint, split);
        engine.ForwardInterceptor = memo.Forward;
        return new Scope(engine);
    }

    private sealed class Scope(LayaEngine engine) : IDisposable
    {
        public void Dispose() => engine.ForwardInterceptor = null;
    }
}
