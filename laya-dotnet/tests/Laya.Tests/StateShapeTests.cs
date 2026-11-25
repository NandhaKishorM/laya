using System.Collections;
using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// <c>lang._iter_text</c> walks mappings, lists and tuples and nothing else (so a set, a generator or any other
/// iterable holds no text), and <c>json.dumps</c> refuses a set. Expected values: <c>state_text(...)</c> in
/// CPython 3.12 for <c>{"a": {"x", "y"}}</c> (""), <c>["a", "b"]</c> and <c>("a", "b")</c> ("a b"),
/// <c>{"k": ["v", ("w",)]}</c> ("v w"), and a generator ("").
/// </summary>
public sealed class StateShapeTests
{
    private sealed class OnlyReadOnlyDictionary : IReadOnlyDictionary<string, object?>
    {
        private readonly Dictionary<string, object?> _inner = new() { ["m"] = "mapped", ["n"] = new[] { "list" } };
        public object? this[string key] => _inner[key];
        public IEnumerable<string> Keys => _inner.Keys;
        public IEnumerable<object?> Values => _inner.Values;
        public int Count => _inner.Count;
        public bool ContainsKey(string key) => _inner.ContainsKey(key);
        public bool TryGetValue(string key, out object? value) => _inner.TryGetValue(key, out value);
        public IEnumerator<KeyValuePair<string, object?>> GetEnumerator() => _inner.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    [Fact]
    public void ASetAndALazySequenceHoldNoText()
    {
        Assert.Equal("", LanguageDetection.StateText(new Dictionary<string, object?> { ["a"] = new HashSet<string> { "x", "y" } }));
        Assert.Equal("", LanguageDetection.StateText(new SortedSet<string> { "x" }));
        Assert.Equal("", LanguageDetection.StateText(Enumerable.Range(0, 3).Select(i => "t" + i)));
        Assert.Equal("", LanguageDetection.StateText(new Queue<string>(["q"])));
    }

    [Fact]
    public void ArraysListsAndTuplesAreWalked()
    {
        Assert.Equal("a b", LanguageDetection.StateText(new[] { "a", "b" }));
        Assert.Equal("a b", LanguageDetection.StateText(new List<object?> { "a", "b" }));
        Assert.Equal("a b", LanguageDetection.StateText(("a", "b")));
        Assert.Equal("v w", LanguageDetection.StateText(new Dictionary<string, object?> { ["k"] = new object[] { "v", ("w", 1) } }));
    }

    [Fact]
    public void AnyMappingTypeIsWalkedByItsValues() =>
        Assert.Equal("mapped list", LanguageDetection.StateText(new OnlyReadOnlyDictionary()));

    [Fact]
    public void ASetIsNotJsonSerializableButAListIs()
    {
        Assert.Throws<JsonException>(() => PythonJson.State(new HashSet<string> { "x" }));
        Assert.Throws<JsonException>(() => PythonJson.State(Enumerable.Range(0, 2)));
        Assert.Equal("[\"a\", 1]", PythonJson.State(new object[] { "a", 1 }));
        Assert.Equal("[\"a\", 1]", PythonJson.State(("a", 1)));
    }
}
