namespace Laya;

/// <summary>The three question types Laya answers. Values match Python's <c>QTYPES</c>.</summary>
public enum QuestionType
{
    /// <summary>Pick one label from a set of named options.</summary>
    Choice = 0,

    /// <summary>Place the state on an ordered scale; the answer is an expected level.</summary>
    Score = 1,

    /// <summary>Does the statement hold? The answer is a probability.</summary>
    Noul = 2,
}

/// <summary>
/// One typed question. Construct with <see cref="Choice(object, ValueTuple{string, object}[])"/>,
/// <see cref="Score(object, object[])"/> or <see cref="Noul(object, object, object)"/>.
/// </summary>
/// <remarks>
/// <para>
/// <c>Instructions</c> is <see cref="object"/> rather than <see cref="string"/> on purpose: Python
/// accepts a non-string and runs it through <c>json.dumps</c>, which escapes non-ASCII, so the
/// rendered prompt differs from the pass-through a string gets. Both paths are reproduced.
/// </para>
/// <para>
/// Criteria order is load-bearing. Python reads choice labels positionally
/// (<c>keys = list(q["crit"].keys())</c>), so the Nth option's probability is attached to the Nth
/// label. Every criteria collection here therefore preserves insertion order.
/// </para>
/// </remarks>
public abstract class Question
{
    internal Question(QuestionType type, object instructions)
    {
        ArgumentNullException.ThrowIfNull(instructions);
        Type = type;
        Instructions = instructions;
    }

    /// <summary>Which of the three question types this is.</summary>
    public QuestionType Type { get; }

    /// <summary>
    /// The prompt. A <see cref="string"/> is used verbatim; anything else is serialized with
    /// <see cref="PythonJson.Instructions"/>, matching Python's <c>_to_internal</c>.
    /// </summary>
    public object Instructions { get; }

    /// <summary>A choice question over ordered <c>(label, description)</c> options.</summary>
    /// <remarks>A null or empty description means "no description", exactly as in Python; note
    /// that <c>0</c> and <c>false</c> are real descriptions and are rendered.</remarks>
    public static ChoiceQuestion Choice(object instructions, params (string Label, object? Description)[] options)
        => new(instructions, options.Select(o => new KeyValuePair<string, object?>(o.Label, o.Description)));

    /// <summary>A choice question over bare labels, with no descriptions.</summary>
    public static ChoiceQuestion Choice(object instructions, params string[] labels)
        => new(instructions, labels.Select(l => new KeyValuePair<string, object?>(l, null)));

    /// <summary>A choice question from any ordered label/description sequence.</summary>
    public static ChoiceQuestion Choice(object instructions, IEnumerable<KeyValuePair<string, object?>> options)
        => new(instructions, options);

    /// <summary>A score question over ordered levels, lowest first.</summary>
    public static ScoreQuestion Score(object instructions, params object?[] levels)
        => new(instructions, levels);

    /// <summary>A score question over ordered levels, lowest first.</summary>
    public static ScoreQuestion Score(object instructions, IEnumerable<object?> levels)
        => new(instructions, levels);

    /// <summary>
    /// A yes/no question. Supplying <paramref name="ifFalse"/> / <paramref name="ifTrue"/> replaces
    /// the default "no, the statement does not hold" / "yes, the statement holds" wording.
    /// </summary>
    public static NoulQuestion Noul(object instructions, object? ifFalse = null, object? ifTrue = null)
        => new(instructions, ifFalse, ifTrue);
}

/// <summary>A <see cref="QuestionType.Choice"/> question: ordered, named options.</summary>
public sealed class ChoiceQuestion : Question
{
    internal ChoiceQuestion(object instructions, IEnumerable<KeyValuePair<string, object?>> options)
        : base(QuestionType.Choice, instructions)
    {
        ArgumentNullException.ThrowIfNull(options);
        var list = new List<KeyValuePair<string, object?>>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var o in options)
        {
            ArgumentNullException.ThrowIfNull(o.Key, nameof(options));
            // Python would silently collapse duplicate keys into one option, changing the option
            // count and so every marker position. Refuse instead of answering a different question.
            if (!seen.Add(o.Key))
                throw new ArgumentException($"duplicate choice label '{o.Key}'", nameof(options));
            list.Add(o);
        }
        if (list.Count == 0)
            throw new ArgumentException("a choice question needs at least one option", nameof(options));
        Options = list;
    }

    /// <summary>The options in label order. Index N here is the label for logit N.</summary>
    public IReadOnlyList<KeyValuePair<string, object?>> Options { get; }

    /// <summary>The labels, in order.</summary>
    public IReadOnlyList<string> Labels => Options.Select(o => o.Key).ToList();
}

/// <summary>A <see cref="QuestionType.Score"/> question: an ordered scale, lowest level first.</summary>
public sealed class ScoreQuestion : Question
{
    internal ScoreQuestion(object instructions, IEnumerable<object?> levels)
        : base(QuestionType.Score, instructions)
    {
        ArgumentNullException.ThrowIfNull(levels);
        Levels = levels.ToList();
        if (Levels.Count == 0)
            throw new ArgumentException("a score question needs at least one level", nameof(levels));
    }

    /// <summary>The levels, lowest first. The answer is an expectation over these indices.</summary>
    public IReadOnlyList<object?> Levels { get; }
}

/// <summary>A <see cref="QuestionType.Noul"/> question: does the statement hold?</summary>
public sealed class NoulQuestion : Question
{
    internal NoulQuestion(object instructions, object? ifFalse, object? ifTrue)
        : base(QuestionType.Noul, instructions)
    {
        IfFalse = ifFalse;
        IfTrue = ifTrue;
    }

    /// <summary>Wording for the false option, or null for the default.</summary>
    public object? IfFalse { get; }

    /// <summary>Wording for the true option, or null for the default.</summary>
    public object? IfTrue { get; }
}

/// <summary>
/// An insertion-ordered set of question id to <see cref="Question"/>, answered in one batch.
/// </summary>
/// <remarks>
/// Deliberately not a <see cref="Dictionary{TKey,TValue}"/>: the batch row order must match the
/// order the caller wrote the questions in, because that is what Python's <c>dict</c> guarantees
/// and what the answer order is compared against.
/// </remarks>
public sealed class QuestionSet : IEnumerable<KeyValuePair<string, Question>>
{
    private readonly List<string> _order = [];
    private readonly Dictionary<string, Question> _byId = new(StringComparer.Ordinal);

    /// <summary>An empty set.</summary>
    public QuestionSet() { }

    /// <summary>A set seeded from an ordered sequence.</summary>
    public QuestionSet(IEnumerable<KeyValuePair<string, Question>> questions)
    {
        ArgumentNullException.ThrowIfNull(questions);
        foreach (var (id, q) in questions) this[id] = q;
    }

    /// <summary>How many questions are in the set.</summary>
    public int Count => _order.Count;

    /// <summary>The question ids, in insertion order.</summary>
    public IReadOnlyList<string> Ids => _order;

    /// <summary>Get or add a question by id. Re-assigning an existing id keeps its position.</summary>
    public Question this[string id]
    {
        get => _byId[id];
        set
        {
            ArgumentException.ThrowIfNullOrEmpty(id);
            ArgumentNullException.ThrowIfNull(value);
            if (!_byId.ContainsKey(id)) _order.Add(id);
            _byId[id] = value;
        }
    }

    /// <summary>Append a question. Throws if the id is already present.</summary>
    public void Add(string id, Question question)
    {
        ArgumentException.ThrowIfNullOrEmpty(id);
        ArgumentNullException.ThrowIfNull(question);
        if (!_byId.TryAdd(id, question))
            throw new ArgumentException($"duplicate question id '{id}'", nameof(id));
        _order.Add(id);
    }

    /// <summary>Whether the set contains this id.</summary>
    public bool ContainsId(string id) => _byId.ContainsKey(id);

    /// <inheritdoc/>
    public IEnumerator<KeyValuePair<string, Question>> GetEnumerator()
    {
        foreach (var id in _order) yield return new(id, _byId[id]);
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
}
