using System.Text;
using System.Text.RegularExpressions;

namespace Laya;

/// <summary>Email utilities for cleaning and structuring email inputs. Ports <c>laya/email.py</c>.</summary>
public static partial class LayaEmail
{
    // re.match anchors at the start only; every pattern below carries its own leading `^`, so
    // .NET's Regex.IsMatch (which is not implicitly start-anchored) reproduces that exactly.
    // `re.I` -> RegexOptions.IgnoreCase.
    [GeneratedRegex(@"^\s*On .{0,300}wrote:\s*$", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderOnWrote();

    [GeneratedRegex(@"^\s*(Em (?=.*\d).{0,300}escreveu:|El (?=.*\d).{0,300}escribi[óo]:|Le (?=.*\d).{0,300}a [eé]crit\s*:)\s*$", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderTranslated();

    [GeneratedRegex(@"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderOriginalOrForwarded();

    [GeneratedRegex(@"^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado)|Message d'origine)\s*-{2,}", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderTranslatedMessage();

    [GeneratedRegex(@"^\s*_{8,}\s*$")]
    private static partial Regex QuoteHeaderUnderscores();

    [GeneratedRegex(@"^\s*(From:|De\s*:)\s.*[@<]", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderFromLine();

    [GeneratedRegex(@"^\s*(De|From)\s*:\s+\S", RegexOptions.IgnoreCase)]
    private static partial Regex HeaderFromName();

    [GeneratedRegex(@"^\s*(Enviad[oa]( em| el)?:\s|Envoy[ée]( le)?\s*:\s|Sent:\s|(Data|Fecha|Date):\s.*\d{4})", RegexOptions.IgnoreCase)]
    private static partial Regex HeaderNext();

    [GeneratedRegex(@"^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]|a [eé]crit)\s*:\s*$", RegexOptions.IgnoreCase)]
    private static partial Regex AttributionTail();

    [GeneratedRegex(@"^\s*(On|Em|El|Le) (?=.*\d)", RegexOptions.IgnoreCase)]
    private static partial Regex AttributionHead();

    [GeneratedRegex(@"^\s*--\s*$")]
    private static partial Regex SignatureDashDash();

    [GeneratedRegex(@"^\s*(?i:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)"
        + @"(?i:\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?")]
    private static partial Regex SignatureClosingHead();

    [GeneratedRegex(@"^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?"
        + @"( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|"
        + @"(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?|"
        + @"(bien )?cordialement|salutations( distinguées)?|bien à vous|merci( d'avance)?|bonne journée)[\s,!.]*$",
        RegexOptions.IgnoreCase)]
    private static partial Regex SignatureTranslated();

    private const string Device = @"iphone|ipad|android|ios|mobile|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|outlook|yahoo|mail|e-?mail|gmail|windows";

    [GeneratedRegex(@"^\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?|"
        + @"envoy[ée] (depuis|de)( mon| ma| mes)?)"
        + @" (" + Device + @")( (" + Device + @"|para|for|no|na|\d+|phone|device|pro|max|mini|plus|using [a-z][a-z0-9_.+-]*))*"
        + @"|(obter o|get) outlook (para|for) (ios|android))[\s.!]*$", RegexOptions.IgnoreCase)]
    private static partial Regex SignatureSentFromMobile();

    // .search() semantics: no anchors, matches anywhere in the text.
    [GeneratedRegex(
        @"(\b(e-?mail|message|information|communication|transmission|contents?)\b[^.]{0,60}"
        + @"\bconfidential\b[^.]{0,60}\b(intended|solely|addressee|recipient|privileged|disclos|unauthori[sz]ed)|"
        + @"\bconfidential\b[^.]{0,60}\b(and (may|is) (also )?privileged)|"
        + @"if you (have )?received this (e-?mail|message) in error|"
        + @"\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|"
        + @"\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}(destinatári|destinatari|pessoa|persona|entidade|entidad)|"
        + @"\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|"
        + @"\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error|"
        + @"\bantes de imprimir\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|"
        + @"\b(meio|medio) ambiente\b[^.]{0,30}antes de imprimir|"
        + @"\b(ce|cet|cette) (message|e-?mail|mail|courriel)\b[^.]{0,80}(confidentiel|privil[eé]gi)|"
        + @"\bavez re[çc]u (ce|cet|cette) (message|e-?mail|mail)\b[^.]{0,20} par erreur|"
        + @"\b(usage exclusif|exclusivement|uniquement)\b[^.]{0,30}destinataire)",
        RegexOptions.IgnoreCase)]
    private static partial Regex Disclaimer();

    // Sentence boundary: split just after a `.`/`!`/`?` and the whitespace that follows it. .NET
    // Regex supports the lookbehind natively, so this needs no hand-written splitter.
    [GeneratedRegex(@"(?<=[.!?])\s+")]
    private static partial Regex SentenceBoundary();

    [GeneratedRegex(@"\n\s*\n")]
    private static partial Regex ParagraphBreak();

    [GeneratedRegex(@"[ \t]+")]
    private static partial Regex SpaceTabRun();

    private static readonly Regex[] QuoteHeaders =
    [
        QuoteHeaderOnWrote(), QuoteHeaderTranslated(), QuoteHeaderOriginalOrForwarded(),
        QuoteHeaderTranslatedMessage(), QuoteHeaderUnderscores(), QuoteHeaderFromLine(),
    ];

    private static readonly Regex[] SignatureMarkers =
    [
        SignatureDashDash(), SignatureTranslated(),
    ];

    /// <summary>Remove quoted email history, signatures and disclaimers to keep input focused.</summary>
    public static string CleanBody(string? body, int maxChars = 3000)
    {
        var text = (body ?? string.Empty).Replace("\r\n", "\n").Replace("\r", "\n").Replace("\\n", "\n");
        if (CodePointLength(text) > maxChars * 4)
            text = LanguageDetection.SliceCodePoints(text, maxChars * 4);

        var lines = new List<string>();
        var source = text.Split('\n');
        for (var i = 0; i < source.Length; i++)
        {
            var line = source[i];
            if (lines.Count > 0 && Array.Exists(QuoteHeaders, p => p.IsMatch(line))) break;
            if (lines.Count > 0 && HeaderFromName().IsMatch(line) && i + 1 < source.Length
                && HeaderNext().IsMatch(source[i + 1])) break;
            if (lines.Count > 0 && AttributionTail().IsMatch(line))
            {
                if (AttributionHead().IsMatch(lines[^1])) lines.RemoveAt(lines.Count - 1);
                break;
            }
            if (line.TrimStart().StartsWith('>')) continue;
            lines.Add(line.TrimEnd());
        }

        var cut = lines.Count;
        var start = Math.Max(1, Math.Min((int)(lines.Count * 0.6), lines.Count - 8));
        for (var i = start; i < lines.Count; i++)
        {
            var stripped = lines[i].Trim();
            var length = CodePointLength(stripped);
            if ((length <= 40 && (IsEnglishSignoff(lines[i]) || Array.Exists(SignatureMarkers, p => p.IsMatch(lines[i]))))
                || (length <= 60 && SignatureSentFromMobile().IsMatch(lines[i])))
            {
                cut = i;
                break;
            }
        }
        lines = lines.GetRange(0, cut);

        var paragraphs = ParagraphBreak().Split(string.Join('\n', lines)).Select(StripDisclaimer);
        var collapsed = string.Join("\n\n", paragraphs.Select(p => p.Trim()).Where(p => p.Length > 0));
        text = SpaceTabRun().Replace(collapsed, " ");
        return LanguageDetection.SliceCodePoints(text, maxChars);
    }

    private static bool IsEnglishSignoff(string line)
    {
        var head = SignatureClosingHead().Match(line);
        if (!head.Success) return false;
        var tail = new List<Rune>();
        foreach (var rune in line[head.Length..].EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) is System.Globalization.UnicodeCategory.NonSpacingMark
                or System.Globalization.UnicodeCategory.SpacingCombiningMark
                or System.Globalization.UnicodeCategory.EnclosingMark
                && tail.Count > 0 && !Rune.IsWhiteSpace(tail[^1])) continue;
            tail.Add(rune);
        }
        var i = 0;
        while (i < tail.Count && (Rune.IsWhiteSpace(tail[i]) || tail[i].Value is ',' or ';' or ':' or '!' or '.')) i++;
        var tokens = 0;
        while (i < tail.Count)
        {
            if (++tokens > 3 || Rune.GetUnicodeCategory(tail[i]) is not
                (System.Globalization.UnicodeCategory.UppercaseLetter
                or System.Globalization.UnicodeCategory.TitlecaseLetter
                or System.Globalization.UnicodeCategory.OtherLetter)) return false;
            i++;
            while (i < tail.Count && (Rune.IsLetterOrDigit(tail[i])
                || Rune.GetUnicodeCategory(tail[i]) is System.Globalization.UnicodeCategory.LetterNumber
                    or System.Globalization.UnicodeCategory.OtherNumber
                || tail[i].Value is '_' or '\'' or '-')) i++;
            while (i < tail.Count && (Rune.IsWhiteSpace(tail[i]) || tail[i].Value is ',' or '.')) i++;
        }
        return true;
    }

    /// <summary>
    /// Drop boilerplate disclaimer text from one paragraph.
    /// </summary>
    /// <remarks>
    /// A paragraph is dropped whole only when *every* sentence in it is boilerplate; otherwise
    /// only the boilerplate sentences go. A footer that runs on without a blank line would
    /// otherwise take the sender's actual request with it, which is worse than leaving one
    /// boilerplate line behind.
    /// </remarks>
    private static string StripDisclaimer(string paragraph)
    {
        if (!Disclaimer().IsMatch(paragraph)) return paragraph; // nothing to do: keep the original line structure
        var pieces = new List<string>();
        foreach (var part in SentenceBoundary().Split(paragraph).Select(p => p.Trim()).Where(p => p.Length > 0))
        {
            if (!Disclaimer().IsMatch(part)) { pieces.Add(part); continue; }
            var buffer = "";
            foreach (var line in part.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0))
            {
                var firstLetter = line.EnumerateRunes().FirstOrDefault(Rune.IsLetter);
                if (buffer.Length > 0 && Rune.IsUpper(firstLetter))
                {
                    pieces.Add(buffer);
                    buffer = line;
                }
                else buffer = buffer.Length > 0 ? buffer + " " + line : line;
            }
            if (buffer.Length > 0) pieces.Add(buffer);
        }
        return string.Join(' ', pieces.Where(p => !Disclaimer().IsMatch(p)));
    }

    /// <summary>Construct a clean state dictionary for email classification.</summary>
    /// <param name="subject">The email subject. Whitespace-trimmed regardless of <paramref name="clean"/>.</param>
    /// <param name="body">The email body.</param>
    /// <param name="sender">Optional sender, stored under the key <c>"from"</c> when non-empty.</param>
    /// <param name="clean">
    /// Whether <paramref name="body"/> is run through <see cref="CleanBody"/>. When
    /// <see langword="false"/>, the raw body is kept as-is.
    /// </param>
    /// <param name="extra">
    /// Additional fields to merge in, in the given order; a <see langword="null"/> value is
    /// dropped entirely rather than stored, and a key that collides with <c>subject</c>/<c>body</c>/
    /// <c>from</c> overwrites it in place, matching Python's <c>**extra</c> plus <c>dict.update</c>.
    /// </param>
    public static IReadOnlyDictionary<string, object?> State(
        string? subject, string? body, string? sender = null, bool clean = true,
        IEnumerable<KeyValuePair<string, object?>>? extra = null)
    {
        var state = new OrderedDictionary<string, object?>(StringComparer.Ordinal)
        {
            ["subject"] = (subject ?? string.Empty).Trim(),
            ["body"] = clean ? CleanBody(body) : body ?? string.Empty,
        };
        if (!string.IsNullOrEmpty(sender)) state["from"] = sender;
        if (extra is not null)
            foreach (var (key, value) in extra)
                if (value is not null) state[key] = value;
        return state;
    }

    private static int CodePointLength(string s)
    {
        var count = 0;
        foreach (var _ in s.EnumerateRunes()) count++;
        return count;
    }
}
