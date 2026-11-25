namespace Laya;

/// <summary>
/// A caller-supplied answer to "can the English checkpoint read this state?" -- Python's
/// <c>lang_guess</c>: a language code, or a callback taking the state.
/// </summary>
/// <remarks>
/// <para>
/// It is an opt-in hint, checked after an explicit <c>lang</c> and before the built-in detection.
/// Any code that is not English routes to the multilingual checkpoint. A code that names no
/// language (blank, <c>C</c>, <c>POSIX</c>, <c>C.UTF-8</c>, <c>und</c>, <c>zxx</c>, <c>mul</c>), or
/// a callback returning <see langword="null"/>, abstains, and routing falls through to detection;
/// that is how a language-identification model that is not confident says so.
/// </para>
/// <para>
/// Pass one per call (the <c>langGuess</c> overloads of <see cref="LayaRouter.Route(object?, QuestionSet?, string?, string?, string?, LangGuess?)"/>)
/// or install one for every request with <see cref="LayaRouterOptions.LangGuess"/>.
/// </para>
/// </remarks>
public sealed class LangGuess
{
    private readonly string? _code;
    private readonly Func<object?, string?>? _callback;

    private LangGuess(string? code, Func<object?, string?>? callback)
    {
        _code = code;
        _callback = callback;
    }

    /// <summary>A hint that is a language code such as <c>"en"</c>, <c>"en-US"</c>, <c>"en_US.UTF-8"</c> or <c>"pt-BR"</c>.</summary>
    public static LangGuess FromCode(string? code) => new(code, null);

    /// <summary>
    /// A hint that reads the state and answers with a language code, or <see langword="null"/> to abstain.
    /// </summary>
    public static LangGuess FromCallback(Func<object?, string?> callback)
    {
        ArgumentNullException.ThrowIfNull(callback);
        return new(null, callback);
    }

    /// <summary>A code string converts to a code hint.</summary>
    public static implicit operator LangGuess(string code) => FromCode(code);

    /// <summary>Python's <c>_resolve_hint</c>: English / not English, or <see langword="null"/> when the hint names nothing.</summary>
    internal bool? Resolve(object? state) =>
        EnglishFromCode(_callback is null ? _code : _callback(state));

    // Codes that are valid $LANG values but name no language, so they answer nothing about the
    // state: router.py:242 _LANGUAGE_AGNOSTIC_CODES.
    private static readonly string[] LanguageAgnosticCodes = ["c", "posix", "und", "zxx", "mul"];

    /// <summary>
    /// Python's <c>_english_from_code</c> (router.py:245): <see langword="true"/> for an English code,
    /// <see langword="false"/> for any other language, <see langword="null"/> when the code identifies nothing.
    /// Accepts <c>"en"</c>, <c>"EN"</c>, <c>"en-US"</c>, <c>"en_US"</c> and <c>"en_US.UTF-8"</c>.
    /// </summary>
    internal static bool? EnglishFromCode(string? value)
    {
        if (value is null) return null;
        var code = LanguageDetection.PythonLower(PythonStr.Strip(value));
        if (code.Length == 0) return null;
        code = code.Split('.', 2)[0];                                 // en_US.UTF-8 -> en_US
        var primary = code.Replace('_', '-').Split('-', 2)[0];        // en_US -> en
        if (primary.Length == 0 || Array.IndexOf(LanguageAgnosticCodes, primary) >= 0) return null;
        return primary is "en" or "eng" or "english";
    }
}
