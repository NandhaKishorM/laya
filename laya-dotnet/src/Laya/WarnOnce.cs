using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Laya;

/// <summary>
/// Marks a public entry whose <em>caller</em> is the call site of any warning raised below it: Python's
/// <c>warnings.warn(..., stacklevel=2 or 3)</c> attributes a warning to the code that called the public
/// method, not to the library line that raised it. Such a method must also be
/// <c>[MethodImpl(MethodImplOptions.NoInlining)]</c> so its frame exists in a Release build.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
internal sealed class WarnSiteAttribute : Attribute;

/// <summary>
/// Python's default warnings filter shows an identical warning once per (text, category, module, line) of
/// the frame <c>stacklevel</c> selects. The category is always <c>RuntimeWarning</c> here; the frame is found by
/// walking the stack to the nearest <see cref="WarnSiteAttribute"/> entry and taking its caller, keyed by
/// that caller's method and IL offset, which tells call sites apart exactly without a PDB. Reached through
/// <see cref="LayaRouter"/> the caller is the router's own call, as Python attributes it to <c>router.py</c>.
/// </summary>
internal static class WarnOnce
{
    private static readonly ConcurrentDictionary<(string Site, string Text), byte> Seen = new();

    /// <summary>Trace <paramref name="text"/> as a warning unless this call site already did. Thread-safe.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static void Warn(string text)
    {
        if (Seen.TryAdd((Site(), text), 0))
            Trace.TraceWarning("{0}", text);
    }

    /// <summary>Forget every warning shown so far (tests only).</summary>
    internal static void Reset() => Seen.Clear();

    private static bool IsEntry(StackFrame frame) =>
        frame.GetMethod() is { } method && method.IsDefined(typeof(WarnSiteAttribute), false);

    // Library plumbing between a public entry and the warning; with no entry on the stack (a config parsed
    // or a window budgeted directly) the first frame outside it is the site.
    private static bool IsPlumbing(StackFrame frame) =>
        frame.GetMethod()?.DeclaringType is { } type
        && (type == typeof(WarnOnce) || type == typeof(LayaConfig) || type == typeof(LongPrediction) || type == typeof(LangTemperatures) || type == typeof(CalibrationFile));

    private static string Site()
    {
        var frames = new StackTrace(false).GetFrames();
        var i = 0;
        while (i < frames.Length && !IsEntry(frames[i])) i++;
        if (i < frames.Length)
            while (i < frames.Length && IsEntry(frames[i])) i++;   // overloads that call each other are one entry
        else
        {
            i = 0;
            while (i < frames.Length && IsPlumbing(frames[i])) i++;
        }
        if (i >= frames.Length || frames[i].GetMethod() is not { } method) return "";
        return $"{method.Module.ModuleVersionId}:{method.MetadataToken}:{frames[i].GetILOffset()}";
    }
}
