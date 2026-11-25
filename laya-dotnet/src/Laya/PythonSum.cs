namespace Laya;

/// <summary>
/// Python's builtin <c>sum()</c> over floats as CPython 3.12 computes it (<c>builtin_sum_impl</c> in
/// <c>Python/bltinmodule.c</c>): Neumaier compensated summation, not a plain left-to-right add. The goldens
/// come from CPython 3.12, so <c>jev_confidence</c> is only bit-identical with this.
/// </summary>
internal static class PythonSum
{
    /// <summary>
    /// <c>sum(values)</c> of floats. Start at <c>0.0</c> with compensation <c>c = 0.0</c>; for each <c>x</c>:
    /// <c>t = f + x</c>; if <c>|f| &gt;= |x|</c> then <c>c += (f - t) + x</c> else <c>c += (x - t) + f</c>; <c>f = t</c>.
    /// At the end, <c>f + c</c> only when <c>c</c> is nonzero and finite (so an infinite or overflowed sum never
    /// becomes NaN), else <c>f</c>. (CPython adds the first float to the int start <c>0</c> uncompensated,
    /// which equals this loop's first step.)
    /// </summary>
    internal static double Sum(IEnumerable<double> values)
    {
        var f = 0.0;
        var c = 0.0;
        foreach (var x in values)
        {
            var t = f + x;
            c += Math.Abs(f) >= Math.Abs(x) ? (f - t) + x : (x - t) + f;
            f = t;
        }
        return c != 0.0 && double.IsFinite(c) ? f + c : f;
    }
}
