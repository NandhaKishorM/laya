using Laya.Tokenization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Laya;

/// <summary>
/// Answers typed questions about a piece of state in one forward pass. The .NET counterpart of
/// Python's <c>laya.Agent</c>: this library is a .NET port of the Laya Python SDK (the
/// <c>laya</c> package on PyPI), run on ONNX Runtime instead of PyTorch.
/// </summary>
/// <remarks>
/// <para>
/// Loading the model is expensive and the session is reusable, so create one engine and keep it:
/// it is safe to share across threads and works well as a singleton.
/// </para>
/// <para>
/// Every question in a <see cref="QuestionSet"/> is answered by a single batched inference call,
/// matching Python's single forward pass — asking five questions costs roughly what asking one does.
/// </para>
/// <para>
/// Loads either ONNX artifact layout (<see cref="LayaArtifactLayout"/>). For the fused layout,
/// one <see cref="InferenceSession"/> runs the whole forward pass. For the split layout (the
/// laya-ts exporter's output), two sessions run in sequence: the encoder produces
/// <c>last_hidden_state</c>, which is fed straight into the head. Both cases go through
/// <see cref="Run"/>, so batching, calibration and <see cref="Predict"/> do not know which
/// layout is loaded.
/// </para>
/// </remarks>
public sealed class LayaEngine : IDisposable, ILayaPredictor
{
    private const string ModelName = "laya-rl-agent";

    // Exactly one of _fusedSession or (_encoderSession, _headSession) is non-null, matching
    // Artifacts.Layout.
    private readonly InferenceSession? _fusedSession;
    private readonly InferenceSession? _encoderSession;
    private readonly InferenceSession? _headSession;
    private readonly ILayaTokenizer _tokenizer;
    private readonly bool _ownsTokenizer;
    private readonly Lock _tokenizerGate = new();
    // 0 = live, 1 = disposed; int so Interlocked.Exchange can operate on it.
    private int _disposed;

    private LayaEngine(InferenceSession? fusedSession, InferenceSession? encoderSession,
                       InferenceSession? headSession, ILayaTokenizer tokenizer, bool ownsTokenizer,
                       LayaConfig config, ModelArtifacts artifacts)
    {
        _fusedSession = fusedSession;
        _encoderSession = encoderSession;
        _headSession = headSession;
        _tokenizer = tokenizer;
        _ownsTokenizer = ownsTokenizer;
        Config = config;
        Artifacts = artifacts;
    }

    /// <summary>The loaded checkpoint's configuration.</summary>
    public LayaConfig Config { get; }

    /// <summary>Where the loaded artifacts came from.</summary>
    public ModelArtifacts Artifacts { get; }

    /// <summary>
    /// Which checkpoint variant was loaded, or <see langword="null"/> when loaded from an
    /// explicit directory (via <see cref="LayaOptions.ModelDirectory"/> or the
    /// <c>LAYA_ONNX_DIR</c> environment variable) where no checkpoint was inferred.
    /// </summary>
    public LayaCheckpoint? Checkpoint => Artifacts.Checkpoint;

    /// <summary>
    /// Load an engine, resolving the artifact directory from <paramref name="options"/>, the
    /// <c>LAYA_ONNX_DIR</c> environment variable, or the local cache.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// When the loaded config's encoder contradicts the expected encoder for the requested
    /// <see cref="LayaOptions.Checkpoint"/> — indicates the wrong artifact was loaded.
    /// </exception>
    public static LayaEngine Create(LayaOptions? options = null)
    {
        options ??= new LayaOptions();
        var artifacts = ModelArtifacts.Resolve(options);
        var config = LayaConfig.Load(artifacts.ConfigPath);

        // When the checkpoint was resolved via a checkpoint-aware path (LAYA_ONNX_ROOT, cache,
        // or download), validate that the artifact's encoder agrees. A mismatch here means the
        // directory name and the contents disagree — something that happens when a user manually
        // copies or renames a checkpoint directory.
        if (artifacts.Checkpoint is { } expectedCheckpoint)
            ValidateEncoder(config.Encoder, expectedCheckpoint, artifacts.Directory);

        // using disposes sessionOptions on both success and failure; ORT copies the options at
        // CreateSession time so they are not needed after the InferenceSession constructor returns.
        using var sessionOptions = BuildSessionOptions(options);
        InferenceSession? fusedSession = null;
        InferenceSession? encoderSession = null;
        InferenceSession? headSession = null;
        HfTokenizer? tokenizer = null;
        try
        {
            if (artifacts.Layout == LayaArtifactLayout.Fused)
            {
                // The path must point at model.onnx inside its own directory: this checkpoint is
                // in ONNX external-data format and the runtime resolves the ~1.3 GB
                // model.onnx.data sidecar relative to the graph file.
                fusedSession = new InferenceSession(artifacts.ModelPath!, sessionOptions);
            }
            else
            {
                encoderSession = new InferenceSession(artifacts.EncoderPath!, sessionOptions);
                headSession = new InferenceSession(artifacts.HeadPath!, sessionOptions);
            }

            tokenizer = new HfTokenizer(artifacts.TokenizerPath);
            return new LayaEngine(fusedSession, encoderSession, headSession,
                tokenizer, ownsTokenizer: true, config, artifacts);
        }
        catch
        {
            tokenizer?.Dispose();
            fusedSession?.Dispose();
            encoderSession?.Dispose();
            headSession?.Dispose();
            throw;
        }
    }

    /// <summary>Load an engine from an artifact directory containing either ONNX layout.</summary>
    public static LayaEngine FromDirectory(string directory) =>
        Create(new LayaOptions { ModelDirectory = directory });

    private static void ValidateEncoder(string? actualEncoder, LayaCheckpoint checkpoint, string dir)
    {
        // Multilingual uses mmBERT; English and TypedDecisions both use ModernBERT-large.
        // We can only detect a family mismatch (mmBERT vs. ModernBERT), not distinguish English
        // from TypedDecisions, since both share the same encoder string.
        var expectModernBert = checkpoint != LayaCheckpoint.Multilingual;
        var gotModernBert    = actualEncoder is not null
                               && actualEncoder.Contains("ModernBERT", StringComparison.OrdinalIgnoreCase);

        if (expectModernBert != gotModernBert)
        {
            var expected = checkpoint == LayaCheckpoint.Multilingual
                ? "jhu-clsp/mmBERT-base"
                : "answerdotai/ModernBERT-large";
            throw new InvalidOperationException(
                $"Checkpoint '{checkpoint}' expects encoder '{expected}' but the artifact at "
                + $"'{dir}' declares encoder '{actualEncoder ?? "(null)"}'. "
                + "The artifact directory may contain the wrong checkpoint.");
        }
    }

    private static SessionOptions BuildSessionOptions(LayaOptions options)
    {
        var so = new SessionOptions();
        try
        {
            if (options.IntraOpThreads is { } intra) so.IntraOpNumThreads = intra;
            if (options.InterOpThreads is { } inter) so.InterOpNumThreads = inter;
            switch (options.ExecutionProvider)
            {
                case LayaExecutionProvider.Cpu:
                    break;
                case LayaExecutionProvider.Cuda:
                    so.AppendExecutionProvider_CUDA();
                    break;
                case LayaExecutionProvider.DirectMl:
                    so.AppendExecutionProvider_DML();
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(options), options.ExecutionProvider,
                        "unknown execution provider");
            }
            return so;
        }
        catch
        {
            so.Dispose();
            throw;
        }
    }

    /// <summary>Answer every question in <paramref name="questions"/> about <paramref name="state"/>.</summary>
    /// <param name="state">
    /// A <see cref="string"/> is used verbatim; anything else is serialized the way Python's
    /// <c>serialize_state</c> does, so an object or a list of conversation turns both work.
    /// </param>
    /// <param name="questions">The questions to answer, in the order they should be returned.</param>
    /// <exception cref="ArgumentException">
    /// If a question's options cannot fit the head budget, rather than answering a silently
    /// truncated question.
    /// </exception>
    public LayaResult Predict(object? state, QuestionSet questions)
    {
        ArgumentNullException.ThrowIfNull(questions);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) == 1, this);
        if (questions.Count == 0) throw new ArgumentException("no questions to answer", nameof(questions));

        var ids = questions.Ids;
        var items = new List<SequenceItem>(questions.Count);

        // A chronological conversation list is serialized newest-last, so right-truncation
        // (the default) would silently drop the newest turn. Matches Agent._encode_state's
        // `truncate_left = isinstance(state, list)`: only a list-shaped state truncates from the
        // left; string and dict states keep the existing right-truncation.
        var truncateLeft = PythonJson.IsListState(state);

        // The tokenizer wraps a native handle whose thread safety is not documented, so
        // tokenization is serialized. Inference is not: it is the expensive part and
        // InferenceSession.Run is thread-safe.
        lock (_tokenizerGate)
        {
            foreach (var id in ids)
            {
                var question = questions[id];
                var (seq, markers) = SequenceBuilder.Build(
                    _tokenizer, state, question, Config.MaxLen, Config.HeadMaxLen, truncateLeft);

                // A marker dropped for landing past max_len means an option is not in the sequence
                // at all, so its logit would score whatever token happens to sit at position 0.
                if (markers.Length != SequenceBuilder.RenderOptions(question).Count)
                    throw new ArgumentException(
                        $"question '{id}': only {markers.Length} of its {SequenceBuilder.RenderOptions(question).Count} "
                        + $"option markers fit in max_len={Config.MaxLen} with head_max_len={Config.HeadMaxLen} "
                        + "spent on the question; lower head_max_len, raise max_len, or use fewer options", nameof(questions));

                items.Add(new SequenceItem(seq, markers, question.Type));
            }
        }

        var batch = Collator.Collate(items, _tokenizer.PadId);
        var (logits, actLogits) = Run(batch);

        // n_act is the act_logits column count, derived from the tensor's dims[1] rather than
        // hardcoded so the code stays correct if the graph is re-exported with a wider action head.
        var nAct = actLogits.Length / batch.Count;

        var answers = new Dictionary<string, Answer>(questions.Count, StringComparer.Ordinal);
        for (var row = 0; row < ids.Count; row++)
        {
            var question = questions[ids[row]];
            var k = batch.MarkerCounts[row];
            var temperature = Calibration.ResolveTemperature(Config, question.Type, k);
            var rowLogits = logits.AsSpan(row * batch.MarkerCount, batch.MarkerCount);
            var p = Calibration.Probabilities(rowLogits, k, temperature);

            var action = new ActionInfo(
                Calibration.Round4(Calibration.Softmax(actLogits.AsSpan(row * nAct, nAct))[0]));
            answers[ids[row]] = BuildAnswer(question, p, k, action);
        }

        return new LayaResult(ModelName, [.. ids], answers, new Usage(batch.InputTokens, 0));
    }

    private static Answer BuildAnswer(Question question, double[] p, int k, ActionInfo action)
    {
        var confidence = Calibration.Round4(Calibration.ConfidenceFromProbs(p, k));
        var answerConfidence = Calibration.Round4(Calibration.AnswerConfidence(p, k));
        switch (question)
        {
            case ChoiceQuestion choice:
            {
                var labels = choice.Labels;
                var probabilities = new List<KeyValuePair<string, double>>(k);
                // Python zips labels against probabilities, so a mismatch would silently drop the
                // tail rather than misalign; take the shorter for the same reason.
                for (var i = 0; i < Math.Min(k, labels.Count); i++)
                    probabilities.Add(new(labels[i], Calibration.Round4(p[i])));
                return new ChoiceAnswer(labels[Calibration.ArgMax(p)], probabilities,
                                        confidence, answerConfidence, action);
            }

            case ScoreQuestion score:
            {
                var probabilities = new double[p.Length];
                for (var i = 0; i < p.Length; i++) probabilities[i] = Calibration.Round4(p[i]);
                return new ScoreAnswer(Calibration.Round4(Calibration.ExpectedScore(p)),
                                       score.Levels.Select(PythonJson.Criterion).Cast<object?>().ToArray(),
                                       probabilities, confidence, answerConfidence, action);
            }

            default:
            {
                // Noul reports the true-branch probability, and its confidence is the distance from
                // a coin flip rather than the entropy measure the other two types use. Over two
                // options max(p_true, 1 - p_true) equals max(p), so answerConfidence agrees here —
                // matching the Python comment on this exact case in `_decode_answers`.
                var pTrue = p[1];
                return new NoulAnswer(Calibration.Round4(pTrue),
                                      Calibration.Round4(Math.Max(pTrue, 1.0 - pTrue)),
                                      answerConfidence, action);
            }
        }
    }

    /// <summary>One forward pass over a collated batch, returning the raw graph outputs.</summary>
    /// <remarks>
    /// Internal rather than private so the test suite can drive the graph directly — the shape sweep
    /// and the recorded-logits comparison both need the outputs before calibration, and sharing this
    /// session keeps them from loading a second copy of the weights.
    /// </remarks>
    internal (float[] Logits, float[] ActLogits) Run(CollatedBatch batch) =>
        Artifacts.Layout == LayaArtifactLayout.Fused ? RunFused(batch) : RunSplit(batch);

    private (float[] Logits, float[] ActLogits) RunFused(CollatedBatch batch)
    {
        var session = _fusedSession!;
        var inputs = new List<NamedOnnxValue>(5);

        if (session.InputMetadata.ContainsKey("input_ids"))
            inputs.Add(LongInput(session, "input_ids", batch.InputIds, batch.Count, batch.SequenceLength));
        if (session.InputMetadata.ContainsKey("attention_mask"))
            inputs.Add(LongInput(session, "attention_mask", batch.AttentionMask, batch.Count, batch.SequenceLength));
        if (session.InputMetadata.ContainsKey("marker_pos"))
            inputs.Add(LongInput(session, "marker_pos", batch.MarkerPos, batch.Count, batch.MarkerCount));
        if (session.InputMetadata.ContainsKey("marker_mask"))
            inputs.Add(MaskInput(session, "marker_mask", batch.MarkerMask, batch.Count, batch.MarkerCount));
        if (session.InputMetadata.ContainsKey("qtype"))
            inputs.Add(QTypeInput(session, batch.QType, batch.Count));

        using var results = session.Run(inputs);
        return (Output(results, "logits"), Output(results, "act_logits"));
    }

    private (float[] Logits, float[] ActLogits) RunSplit(CollatedBatch batch)
    {
        var encoder = _encoderSession!;
        var head = _headSession!;

        var encoderInputs = new List<NamedOnnxValue>(2);
        if (encoder.InputMetadata.ContainsKey("input_ids"))
            encoderInputs.Add(LongInput(encoder, "input_ids", batch.InputIds, batch.Count, batch.SequenceLength));
        if (encoder.InputMetadata.ContainsKey("attention_mask"))
            encoderInputs.Add(LongInput(encoder, "attention_mask", batch.AttentionMask, batch.Count, batch.SequenceLength));

        // Both Run calls happen inside this one using block: headResults is built and consumed
        // while encoderResults (and the last_hidden_state tensor it owns) is still alive, so the
        // head reads the encoder's output tensor directly with no extra copy.
        using var encoderResults = encoder.Run(encoderInputs);
        var hiddenState = OutputTensor(encoderResults, "last_hidden_state");

        var headInputs = new List<NamedOnnxValue>(5)
        {
            NamedOnnxValue.CreateFromTensor("hidden_states", hiddenState),
        };
        if (head.InputMetadata.ContainsKey("marker_pos"))
            headInputs.Add(LongInput(head, "marker_pos", batch.MarkerPos, batch.Count, batch.MarkerCount));
        if (head.InputMetadata.ContainsKey("marker_mask"))
            headInputs.Add(MaskInput(head, "marker_mask", batch.MarkerMask, batch.Count, batch.MarkerCount));
        if (head.InputMetadata.ContainsKey("qtype"))
            headInputs.Add(QTypeInput(head, batch.QType, batch.Count));
        if (head.InputMetadata.ContainsKey("attention_mask"))
            headInputs.Add(LongInput(head, "attention_mask", batch.AttentionMask, batch.Count, batch.SequenceLength));

        using var headResults = head.Run(headInputs);
        return (Output(headResults, "logits"), Output(headResults, "act_logits"));
    }

    // ── input/output helpers shared by both layouts ────────────────────────────

    /// <summary>
    /// Builds an integer input tensor, matching whichever of int64/int32 the session declares
    /// for <paramref name="name"/> (mmBERT- and ModernBERT-family exports both use int64, but the
    /// declared dtype — not a hardcoded assumption — decides which one is fed).
    /// </summary>
    private static NamedOnnxValue LongInput(InferenceSession session, string name, long[] data, params int[] shape)
    {
        var elementType = session.InputMetadata[name].ElementType;
        if (elementType == typeof(long))
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, shape));
        if (elementType == typeof(int))
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(Array.ConvertAll(data, x => (int)x), shape));

        throw new InvalidOperationException(
            $"input '{name}' is declared as {elementType} in the loaded graph; expected int64 or int32.");
    }

    /// <summary>
    /// Builds a boolean-mask input tensor, matching whichever of bool/float32 the session
    /// declares for <paramref name="name"/>.
    /// </summary>
    private static NamedOnnxValue MaskInput(InferenceSession session, string name, bool[] data, params int[] shape)
    {
        var elementType = session.InputMetadata[name].ElementType;
        if (elementType == typeof(bool))
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<bool>(data, shape));
        if (elementType == typeof(float))
        {
            var floats = new float[data.Length];
            for (var i = 0; i < data.Length; i++) floats[i] = data[i] ? 1f : 0f;
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(floats, shape));
        }

        throw new InvalidOperationException(
            $"input '{name}' is declared as {elementType} in the loaded graph; expected bool or float32.");
    }

    /// <summary>
    /// Builds the <c>qtype</c> input, reading its declared rank from session metadata rather than
    /// hardcoding it per layout: shape <c>[batch]</c> when the graph declares rank 1 (today's
    /// fused export), or <c>[batch, 1]</c> when it declares rank 2 (the split export's head).
    /// </summary>
    private static NamedOnnxValue QTypeInput(InferenceSession session, long[] data, int batchCount)
    {
        var rank = session.InputMetadata["qtype"].Dimensions.Length;
        var shape = rank >= 2 ? new[] { batchCount, 1 } : new[] { batchCount };
        return LongInput(session, "qtype", data, shape);
    }

    private static float[] Output(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, string name)
    {
        foreach (var value in results)
            if (string.Equals(value.Name, name, StringComparison.Ordinal))
                return value.AsEnumerable<float>().ToArray();
        throw new InvalidOperationException($"the model produced no '{name}' output");
    }

    private static Tensor<float> OutputTensor(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, string name)
    {
        foreach (var value in results)
            if (string.Equals(value.Name, name, StringComparison.Ordinal))
                return value.AsTensor<float>();
        throw new InvalidOperationException($"the encoder produced no '{name}' output");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // Interlocked.Exchange is the standard pattern for a one-shot teardown: it atomically
        // sets the flag and returns the old value, so the cleanup runs exactly once even when
        // multiple threads race to dispose.
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        if (_ownsTokenizer && _tokenizer is IDisposable disposable) disposable.Dispose();
        _fusedSession?.Dispose();
        _encoderSession?.Dispose();
        _headSession?.Dispose();
    }
}
