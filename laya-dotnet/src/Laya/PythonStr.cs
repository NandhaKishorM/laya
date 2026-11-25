using System.Globalization;
using System.Text;

namespace Laya;

/// <summary>
/// Python <c>str</c> predicates that .NET spells differently, so the ports of <c>lang.py</c>,
/// <c>router.py</c> and <c>email.py</c> read the same characters as the reference does.
/// </summary>
/// <remarks>
/// The range tables below were generated from CPython 3.12 (Unicode 15.0.0) as
/// <c>unicodedata.combining(ch) != 0</c>, <c>ch.isupper()</c> / <c>ch.islower()</c> for the
/// characters outside <c>Lu</c> / <c>Ll</c>, and <c>ch.isspace()</c>.
/// </remarks>
internal static class PythonStr
{
    private static readonly (int Lo, int Hi)[] CombiningRanges =
    [
        (0x300, 0x34E), (0x350, 0x36F), (0x483, 0x487), (0x591, 0x5BD), (0x5BF, 0x5BF), (0x5C1, 0x5C2),
        (0x5C4, 0x5C5), (0x5C7, 0x5C7), (0x610, 0x61A), (0x64B, 0x65F), (0x670, 0x670), (0x6D6, 0x6DC),
        (0x6DF, 0x6E4), (0x6E7, 0x6E8), (0x6EA, 0x6ED), (0x711, 0x711), (0x730, 0x74A), (0x7EB, 0x7F3),
        (0x7FD, 0x7FD), (0x816, 0x819), (0x81B, 0x823), (0x825, 0x827), (0x829, 0x82D), (0x859, 0x85B),
        (0x898, 0x89F), (0x8CA, 0x8E1), (0x8E3, 0x8FF), (0x93C, 0x93C), (0x94D, 0x94D), (0x951, 0x954),
        (0x9BC, 0x9BC), (0x9CD, 0x9CD), (0x9FE, 0x9FE), (0xA3C, 0xA3C), (0xA4D, 0xA4D), (0xABC, 0xABC),
        (0xACD, 0xACD), (0xB3C, 0xB3C), (0xB4D, 0xB4D), (0xBCD, 0xBCD), (0xC3C, 0xC3C), (0xC4D, 0xC4D),
        (0xC55, 0xC56), (0xCBC, 0xCBC), (0xCCD, 0xCCD), (0xD3B, 0xD3C), (0xD4D, 0xD4D), (0xDCA, 0xDCA),
        (0xE38, 0xE3A), (0xE48, 0xE4B), (0xEB8, 0xEBA), (0xEC8, 0xECB), (0xF18, 0xF19), (0xF35, 0xF35),
        (0xF37, 0xF37), (0xF39, 0xF39), (0xF71, 0xF72), (0xF74, 0xF74), (0xF7A, 0xF7D), (0xF80, 0xF80),
        (0xF82, 0xF84), (0xF86, 0xF87), (0xFC6, 0xFC6), (0x1037, 0x1037), (0x1039, 0x103A), (0x108D, 0x108D),
        (0x135D, 0x135F), (0x1714, 0x1715), (0x1734, 0x1734), (0x17D2, 0x17D2), (0x17DD, 0x17DD), (0x18A9, 0x18A9),
        (0x1939, 0x193B), (0x1A17, 0x1A18), (0x1A60, 0x1A60), (0x1A75, 0x1A7C), (0x1A7F, 0x1A7F), (0x1AB0, 0x1ABD),
        (0x1ABF, 0x1ACE), (0x1B34, 0x1B34), (0x1B44, 0x1B44), (0x1B6B, 0x1B73), (0x1BAA, 0x1BAB), (0x1BE6, 0x1BE6),
        (0x1BF2, 0x1BF3), (0x1C37, 0x1C37), (0x1CD0, 0x1CD2), (0x1CD4, 0x1CE0), (0x1CE2, 0x1CE8), (0x1CED, 0x1CED),
        (0x1CF4, 0x1CF4), (0x1CF8, 0x1CF9), (0x1DC0, 0x1DFF), (0x20D0, 0x20DC), (0x20E1, 0x20E1), (0x20E5, 0x20F0),
        (0x2CEF, 0x2CF1), (0x2D7F, 0x2D7F), (0x2DE0, 0x2DFF), (0x302A, 0x302F), (0x3099, 0x309A), (0xA66F, 0xA66F),
        (0xA674, 0xA67D), (0xA69E, 0xA69F), (0xA6F0, 0xA6F1), (0xA806, 0xA806), (0xA82C, 0xA82C), (0xA8C4, 0xA8C4),
        (0xA8E0, 0xA8F1), (0xA92B, 0xA92D), (0xA953, 0xA953), (0xA9B3, 0xA9B3), (0xA9C0, 0xA9C0), (0xAAB0, 0xAAB0),
        (0xAAB2, 0xAAB4), (0xAAB7, 0xAAB8), (0xAABE, 0xAABF), (0xAAC1, 0xAAC1), (0xAAF6, 0xAAF6), (0xABED, 0xABED),
        (0xFB1E, 0xFB1E), (0xFE20, 0xFE2F), (0x101FD, 0x101FD), (0x102E0, 0x102E0), (0x10376, 0x1037A), (0x10A0D, 0x10A0D),
        (0x10A0F, 0x10A0F), (0x10A38, 0x10A3A), (0x10A3F, 0x10A3F), (0x10AE5, 0x10AE6), (0x10D24, 0x10D27), (0x10EAB, 0x10EAC),
        (0x10EFD, 0x10EFF), (0x10F46, 0x10F50), (0x10F82, 0x10F85), (0x11046, 0x11046), (0x11070, 0x11070), (0x1107F, 0x1107F),
        (0x110B9, 0x110BA), (0x11100, 0x11102), (0x11133, 0x11134), (0x11173, 0x11173), (0x111C0, 0x111C0), (0x111CA, 0x111CA),
        (0x11235, 0x11236), (0x112E9, 0x112EA), (0x1133B, 0x1133C), (0x1134D, 0x1134D), (0x11366, 0x1136C), (0x11370, 0x11374),
        (0x11442, 0x11442), (0x11446, 0x11446), (0x1145E, 0x1145E), (0x114C2, 0x114C3), (0x115BF, 0x115C0), (0x1163F, 0x1163F),
        (0x116B6, 0x116B7), (0x1172B, 0x1172B), (0x11839, 0x1183A), (0x1193D, 0x1193E), (0x11943, 0x11943), (0x119E0, 0x119E0),
        (0x11A34, 0x11A34), (0x11A47, 0x11A47), (0x11A99, 0x11A99), (0x11C3F, 0x11C3F), (0x11D42, 0x11D42), (0x11D44, 0x11D45),
        (0x11D97, 0x11D97), (0x11F41, 0x11F42), (0x16AF0, 0x16AF4), (0x16B30, 0x16B36), (0x16FF0, 0x16FF1), (0x1BC9E, 0x1BC9E),
        (0x1D165, 0x1D169), (0x1D16D, 0x1D172), (0x1D17B, 0x1D182), (0x1D185, 0x1D18B), (0x1D1AA, 0x1D1AD), (0x1D242, 0x1D244),
        (0x1E000, 0x1E006), (0x1E008, 0x1E018), (0x1E01B, 0x1E021), (0x1E023, 0x1E024), (0x1E026, 0x1E02A), (0x1E08F, 0x1E08F),
        (0x1E130, 0x1E136), (0x1E2AE, 0x1E2AE), (0x1E2EC, 0x1E2EF), (0x1E4EC, 0x1E4EF), (0x1E8D0, 0x1E8D6), (0x1E944, 0x1E94A),
    ];

    private static readonly (int Lo, int Hi)[] OtherUpperRanges =
    [
        (0x2160, 0x216F), (0x24B6, 0x24CF), (0x1F130, 0x1F149), (0x1F150, 0x1F169), (0x1F170, 0x1F189),
    ];

    private static readonly (int Lo, int Hi)[] OtherLowerRanges =
    [
        (0xAA, 0xAA), (0xBA, 0xBA), (0x2B0, 0x2B8), (0x2C0, 0x2C1), (0x2E0, 0x2E4), (0x345, 0x345),
        (0x37A, 0x37A), (0x10FC, 0x10FC), (0x1D2C, 0x1D6A), (0x1D78, 0x1D78), (0x1D9B, 0x1DBF), (0x2071, 0x2071),
        (0x207F, 0x207F), (0x2090, 0x209C), (0x2170, 0x217F), (0x24D0, 0x24E9), (0x2C7C, 0x2C7D), (0xA69C, 0xA69D),
        (0xA770, 0xA770), (0xA7F2, 0xA7F4), (0xA7F8, 0xA7F9), (0xAB5C, 0xAB5F), (0xAB69, 0xAB69), (0x10780, 0x10780),
        (0x10783, 0x10785), (0x10787, 0x107B0), (0x107B2, 0x107BA), (0x1E030, 0x1E06D),
    ];

    private static readonly (int Lo, int Hi)[] SpaceRanges =
    [
        (0x9, 0xD), (0x1C, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
        (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000),
    ];


    private static bool InRanges(int cp, (int Lo, int Hi)[] ranges)
    {
        foreach (var (lo, hi) in ranges)
            if (cp >= lo && cp <= hi) return true;
        return false;
    }

    /// <summary><c>ch.isspace()</c>: Unicode White_Space plus U+001C..U+001F, which <c>\s</c> accepts too.</summary>
    internal static bool IsSpace(int cp) => InRanges(cp, SpaceRanges);

    /// <summary><c>unicodedata.combining(ch) != 0</c>: a nonzero canonical combining class.</summary>
    internal static bool IsCombining(int cp) => cp >= 0x300 && InRanges(cp, CombiningRanges);

    /// <summary><c>ch.isupper()</c> for one character.</summary>
    internal static bool IsUpper(Rune r) =>
        Rune.GetUnicodeCategory(r) == UnicodeCategory.UppercaseLetter || InRanges(r.Value, OtherUpperRanges);

    /// <summary><c>ch.islower()</c> for one character.</summary>
    internal static bool IsLower(Rune r) =>
        Rune.GetUnicodeCategory(r) == UnicodeCategory.LowercaseLetter || InRanges(r.Value, OtherLowerRanges);

    private static bool IsTitle(Rune r) => Rune.GetUnicodeCategory(r) == UnicodeCategory.TitlecaseLetter;

    /// <summary><c>s.isupper()</c>: at least one uppercase character and no lowercase or titlecase one.</summary>
    internal static bool StrIsUpper(string s)
    {
        var cased = false;
        foreach (var r in s.EnumerateRunes())
        {
            if (IsLower(r) || IsTitle(r)) return false;
            if (IsUpper(r)) cased = true;
        }
        return cased;
    }

    private static bool IsCased(int cp)
    {
        if (cp is >= 0xD800 and <= 0xDFFF) return false;
        var rune = new Rune(cp);
        return IsUpper(rune) || IsLower(rune) || Rune.GetUnicodeCategory(rune) == UnicodeCategory.TitlecaseLetter;
    }

    // Case_Ignorable: Mn, Me, Cf, Lm, Sk, and the Word_Break MidLetter / MidNumLet / Single_Quote characters.
    private static bool IsCaseIgnorable(int cp)
    {
        if (cp is 0x27 or 0x2E or 0x3A or 0xB7 or 0x387 or 0x55F or 0x5F4 or 0x2018 or 0x2019 or 0x2024 or 0x2027
            or 0xFE13 or 0xFE52 or 0xFE55 or 0xFF07 or 0xFF0E or 0xFF1A) return true;
        if (cp is >= 0xD800 and <= 0xDFFF) return false;
        return Rune.GetUnicodeCategory(new Rune(cp)) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark
            or UnicodeCategory.Format or UnicodeCategory.ModifierLetter or UnicodeCategory.ModifierSymbol;
    }

    /// <summary>
    /// <c>s.lower()</c>: the simple lowercase of every character, U+0130 as <c>i</c> + U+0307, and a capital sigma as
    /// the final form U+03C2 when a cased letter precedes it and none follows (case-ignorable characters skipped).
    /// </summary>
    internal static string Lower(string s)
    {
        var points = new List<int>(s.Length);
        for (var i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                points.Add(char.ConvertToUtf32(s[i], s[i + 1]));
                i++;
            }
            else points.Add(s[i]);
        }
        var sb = new StringBuilder(s.Length);
        for (var i = 0; i < points.Count; i++)
        {
            var cp = points[i];
            if (cp is >= 0xD800 and <= 0xDFFF) sb.Append((char)cp);
            else if (cp == 0x130) sb.Append("i\u0307");
            else if (cp == 0x3A3) sb.Append(FinalSigma(points, i) ? '\u03C2' : '\u03C3');
            else sb.Append(Rune.ToLowerInvariant(new Rune(cp)).ToString());
        }
        return sb.ToString();
    }

    private static bool FinalSigma(List<int> points, int index)
    {
        var j = index - 1;
        while (j >= 0 && IsCaseIgnorable(points[j])) j--;
        if (j < 0 || !IsCased(points[j])) return false;
        j = index + 1;
        while (j < points.Count && IsCaseIgnorable(points[j])) j++;
        return j == points.Count || !IsCased(points[j]);
    }

    /// <summary><c>s.strip()</c>.</summary>
    internal static string Strip(string s)
    {
        var start = 0;
        var end = s.Length;
        while (start < end && IsSpace(s[start])) start++;
        while (end > start && IsSpace(s[end - 1])) end--;
        return s[start..end];
    }

    /// <summary><c>s.rstrip()</c>.</summary>
    internal static string RStrip(string s)
    {
        var end = s.Length;
        while (end > 0 && IsSpace(s[end - 1])) end--;
        return s[..end];
    }

    /// <summary><c>s.lstrip()</c>.</summary>
    internal static string LStrip(string s)
    {
        var start = 0;
        while (start < s.Length && IsSpace(s[start])) start++;
        return s[start..];
    }

    /// <summary><c>s.split()</c> with no argument: runs of whitespace separate, empty pieces dropped.</summary>
    internal static List<string> Split(string s)
    {
        var parts = new List<string>();
        var start = -1;
        for (var i = 0; i < s.Length; i++)
        {
            if (IsSpace(s[i]))
            {
                if (start >= 0) { parts.Add(s[start..i]); start = -1; }
            }
            else if (start < 0) start = i;
        }
        if (start >= 0) parts.Add(s[start..]);
        return parts;
    }

    /// <summary><c>ch.isprintable()</c>: not Other (Cc, Cf, Cs, Co, Cn) or Separator (Zl, Zp, Zs) except the ASCII space.</summary>
    internal static bool IsPrintable(Rune r)
    {
        if (r.Value == ' ') return true;
        return Rune.GetUnicodeCategory(r) is not (UnicodeCategory.Control or UnicodeCategory.Format
            or UnicodeCategory.Surrogate or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned
            or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator or UnicodeCategory.SpaceSeparator);
    }

    /// <summary>Python's <c>repr()</c> of a <see cref="string"/> (and so <c>%r</c>).</summary>
    internal static string Repr(string s)
    {
        var quote = s.Contains('\'') && !s.Contains('"') ? '"' : '\'';
        var sb = new StringBuilder(s.Length + 2);
        sb.Append(quote);
        for (var i = 0; i < s.Length; i++)
        {
            int cp;
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                cp = char.ConvertToUtf32(s[i], s[i + 1]);
                i++;
            }
            else cp = s[i];   // a lone surrogate is not printable, as in Python
            if (cp == quote || cp == '\\') { sb.Append('\\').Append((char)cp); continue; }
            switch (cp)
            {
                case '\n': sb.Append("\\n"); continue;
                case '\r': sb.Append("\\r"); continue;
                case '\t': sb.Append("\\t"); continue;
            }
            if (cp is >= 0xD800 and <= 0xDFFF) sb.Append("\\u").Append(cp.ToString("x4"));
            else if (!IsPrintable(new Rune(cp)))
                sb.Append(cp < 0x100 ? "\\x" + cp.ToString("x2") : cp < 0x10000 ? "\\u" + cp.ToString("x4") : "\\U" + cp.ToString("x8"));
            else sb.Append(char.ConvertFromUtf32(cp));
        }
        sb.Append(quote);
        return sb.ToString();
    }
}
