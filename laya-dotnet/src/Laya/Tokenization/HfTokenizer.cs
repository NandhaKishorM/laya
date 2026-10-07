using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Tokenizers.DotNet;

namespace Laya.Tokenization;

/// <summary>
/// <see cref="ILayaTokenizer"/> backed by the <c>Tokenizers.DotNet</c> native wrapper of the
/// HuggingFace Rust tokenizers library.
/// </summary>
/// <remarks>
/// <para>
/// The <c>tokenizer.json</c> shipped with Laya contains a <c>post_processor</c> of type
/// <c>TemplateProcessing</c> that injects special tokens on every encode:
/// <c>&lt;bos&gt; A &lt;eos&gt;</c> for the multilingual checkpoint and
/// <c>[CLS] A [SEP]</c> for the ModernBERT checkpoints.
/// The Python code calls the tokenizer with <c>add_special_tokens=False</c> on many fragments
/// separately and assembles them itself, so those injected ids would corrupt every downstream
/// position. <c>Tokenizers.DotNet</c> offers no equivalent flag, so this class rewrites the
/// JSON with <c>"post_processor": null</c> before loading it, and caches the result.
/// </para>
/// <para>
/// The special-token ids are read from <c>tokenizer_config.json</c> (which maps role names
/// such as <c>cls_token</c> to token strings) and from <c>tokenizer.json</c>'s
/// <c>added_tokens</c> array; the <c>post_processor.special_tokens</c> section serves as a
/// fallback for tokenizers where not every special token appears in <c>added_tokens</c>.
/// This config-driven approach works for both the mmBERT-style tokens
/// (<c>&lt;pad&gt;</c>, <c>&lt;bos&gt;</c>, …) and the ModernBERT-style tokens
/// (<c>[PAD]</c>, <c>[CLS]</c>, …) without any hard-coding.
/// </para>
/// <para>
/// The split ONNX layout (the laya-ts exporter's output) ships <c>tokenizer.json</c> with no
/// <c>tokenizer_config.json</c> beside it. When that file is missing, each role's token string
/// is instead guessed from conventional special-token spellings (the ModernBERT bracketed form,
/// the SentencePiece/BPE form, and the BOS/EOS convention some tokenizers use for CLS/SEP) and
/// confirmed against <c>tokenizer.json</c>'s own token map before being accepted — see
/// <see cref="FirstKnownToken"/>.
/// </para>
/// </remarks>
public sealed class HfTokenizer : ILayaTokenizer, IDisposable
{
    private readonly Tokenizer _inner;
    private readonly string _maskToken;

    /// <inheritdoc/>
    public int PadId { get; }

    /// <inheritdoc/>
    public int ClsId { get; }

    /// <inheritdoc/>
    public int SepId { get; }

    /// <inheritdoc/>
    public int MaskId { get; }

    /// <inheritdoc/>
    public int UnkId { get; }

    /// <inheritdoc/>
    public string MaskToken => _maskToken;

    /// <summary>
    /// Loads the tokenizer from <paramref name="tokenizerJsonPath"/>, stripping the
    /// <c>post_processor</c> and caching the modified file next to the original (or in a
    /// temp directory when the source directory is read-only).
    /// </summary>
    /// <param name="tokenizerJsonPath">
    /// Absolute or relative path to <c>tokenizer.json</c>. A sibling <c>tokenizer_config.json</c>
    /// is used when present (the fused layout); when absent (the split layout), special-token
    /// strings are instead inferred from convention — see <see cref="FirstKnownToken"/>.
    /// </param>
    /// <exception cref="InvalidOperationException">
    /// When <c>tokenizer_config.json</c> is present but missing a required special-token key, or
    /// a token string is absent from the id maps in <c>tokenizer.json</c> — which typically means
    /// the two files belong to different checkpoints — or, when it is absent, when none of the
    /// conventional spellings for a role appear in <c>tokenizer.json</c> either.
    /// </exception>
    public HfTokenizer(string tokenizerJsonPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(tokenizerJsonPath);

        var strippedPath = EnsureStrippedTokenizer(tokenizerJsonPath);
        _inner = new Tokenizer(vocabPath: strippedPath);

        var tokens = ReadSpecialTokenIds(tokenizerJsonPath);
        PadId      = tokens.Pad;
        ClsId      = tokens.Cls;
        SepId      = tokens.Sep;
        MaskId     = tokens.Mask;
        UnkId      = tokens.Unk;
        _maskToken = tokens.MaskToken;
    }

    /// <inheritdoc/>
    public int[] Encode(string text)
    {
        var tokens = _inner.Encode(text);
        return Array.ConvertAll(tokens, t => (int)t);
    }

    /// <inheritdoc/>
    public void Dispose() => _inner.Dispose();

    // ── post-processor stripping ───────────────────────────────────────────────

    /// <summary>
    /// Returns the path to a cached copy of the tokenizer with <c>post_processor</c> set
    /// to <see langword="null"/>. Regenerates the cache when the source file has changed.
    /// </summary>
    private static string EnsureStrippedTokenizer(string sourcePath)
    {
        sourcePath = Path.GetFullPath(sourcePath);
        var srcInfo = new FileInfo(sourcePath);

        // Fingerprint: source file length and last-write UTC ticks, separated by |.
        // Using ticks avoids locale-dependent date formatting.
        var fingerprint = $"{srcInfo.Length}|{srcInfo.LastWriteTimeUtc.Ticks}";

        // Primary cache: alongside the original.
        var primaryDir = Path.GetDirectoryName(sourcePath)!;
        var primaryPath = Path.Combine(primaryDir, "tokenizer.nopost.json");

        if (IsCacheValid(primaryPath, fingerprint))
            return primaryPath;

        if (TryWriteStripped(sourcePath, primaryPath, fingerprint))
            return primaryPath;

        // Fallback: a per-source temp directory when the artifact directory is read-only.
        var fallbackDir = Path.Combine(
            Path.GetTempPath(), "laya", FingerprintToSafeName(sourcePath));
        System.IO.Directory.CreateDirectory(fallbackDir);

        var fallbackPath = Path.Combine(fallbackDir, "tokenizer.nopost.json");

        if (IsCacheValid(fallbackPath, fingerprint))
            return fallbackPath;

        if (TryWriteStripped(sourcePath, fallbackPath, fingerprint))
            return fallbackPath;

        throw new IOException(
            $"Cannot write stripped tokenizer cache to either:\n" +
            $"  {primaryPath}\n  {fallbackPath}\n" +
            "Check that at least one of those directories is writable.");
    }

    private static bool IsCacheValid(string cachedPath, string fingerprint)
    {
        var metaPath = cachedPath + ".meta";
        if (!File.Exists(cachedPath) || !File.Exists(metaPath))
            return false;
        try { return File.ReadAllText(metaPath).Trim() == fingerprint; }
        catch (IOException) { return false; }
    }

    private static bool TryWriteStripped(string sourcePath, string destPath, string fingerprint)
    {
        try
        {
            WriteStrippedTokenizer(sourcePath, destPath);
            File.WriteAllText(destPath + ".meta", fingerprint);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// Parses <paramref name="sourcePath"/> with <c>JsonNode</c>, sets
    /// <c>"post_processor"</c> to <c>null</c>, and writes the result to
    /// <paramref name="destPath"/> via a <c>.part</c> temp file so an interrupted write
    /// never leaves a partial cache.
    /// </summary>
    private static void WriteStrippedTokenizer(string sourcePath, string destPath)
    {
        JsonObject root;
        using (var stream = File.OpenRead(sourcePath))
            root = JsonNode.Parse(stream)!.AsObject();

        // Setting to null writes "post_processor": null in the output, which the
        // underlying Rust tokenizers library treats identically to the key being absent.
        root["post_processor"] = null;

        var partPath = destPath + ".part";
        using (var outStream = new FileStream(
                   partPath, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            using var writer = new Utf8JsonWriter(outStream);
            root.WriteTo(writer);
        }

        if (File.Exists(destPath)) File.Delete(destPath);
        File.Move(partPath, destPath);
    }

    /// <summary>
    /// Produces a short ASCII directory-name fragment that is stable for a given source
    /// path, used to keep per-source fallback cache directories separated in %TEMP%.
    /// </summary>
    private static string FingerprintToSafeName(string sourcePath)
    {
        // SHA-256 of the UTF-8 path gives a deterministic hash that survives process restarts.
        // string.GetHashCode() is randomised per process in .NET (ASLR seed), so the fallback
        // cache would never hit across runs — exactly the scenario it exists for.
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(sourcePath));
        // First 8 hex chars (4 bytes) are enough for collision resistance within one machine's paths.
        return Convert.ToHexString(hash, 0, 4).ToLowerInvariant();
    }

    // ── special-token ids ──────────────────────────────────────────────────────

    /// <summary>
    /// Reads the special-token ids for the checkpoint whose files are at
    /// <paramref name="tokenizerJsonPath"/>. <c>tokenizer.json</c>'s <c>added_tokens</c> and, as
    /// a fallback, its <c>post_processor.special_tokens</c> supply the id mapping. The token
    /// *strings* normally come from a sibling <c>tokenizer_config.json</c> (the fused layout);
    /// when that file is absent (the split layout, which the laya-ts exporter writes without a
    /// <c>tokenizer_config.json</c>), the strings are instead guessed from conventional special-
    /// token spellings and confirmed against <c>added_tokens</c> — see <see cref="FirstKnownToken"/>.
    /// </summary>
    private static (int Pad, int Cls, int Sep, int Mask, int Unk, string MaskToken)
        ReadSpecialTokenIds(string tokenizerJsonPath)
    {
        tokenizerJsonPath = Path.GetFullPath(tokenizerJsonPath);
        var configPath = Path.Combine(
            Path.GetDirectoryName(tokenizerJsonPath)!, "tokenizer_config.json");

        // 1. Build a content→id map. Primary: added_tokens enumerates all extended-vocabulary
        //    tokens including special ones (both the small mmBERT special-token block at the
        //    front of the vocab, and the ModernBERT block appended at the end). Fallback:
        //    post_processor.special_tokens is also populated by ModernBERT tokenizers and
        //    is useful for any tokenizer that omits a special token from added_tokens. This map
        //    is built before looking at tokenizer_config.json, since the split layout has no
        //    such file and needs added_tokens alone to resolve every role.
        var idByContent = new Dictionary<string, int>(StringComparer.Ordinal);
        string? templateCls = null, templateSep = null;
        using (var f = File.OpenRead(tokenizerJsonPath))
        using (var tok = JsonDocument.Parse(f))
        {
            var r = tok.RootElement;
            foreach (var entry in r.GetProperty("added_tokens").EnumerateArray())
                idByContent[entry.GetProperty("content").GetString()!] =
                    entry.GetProperty("id").GetInt32();

            if (r.TryGetProperty("post_processor", out var pp)
                    && pp.ValueKind == JsonValueKind.Object
                    && pp.TryGetProperty("special_tokens", out var st)
                    && st.ValueKind == JsonValueKind.Object)
            {
                foreach (var kvp in st.EnumerateObject())
                    if (!idByContent.ContainsKey(kvp.Name)
                        && kvp.Value.TryGetProperty("ids", out var ids)
                        && ids.GetArrayLength() > 0)
                        idByContent[kvp.Name] = ids[0].GetInt32();
            }

            // The single-sequence template ("[CLS] $A [SEP]", or "<bos> $A <eos>" for mmBERT)
            // names the CLS and SEP strings outright. Only the split layout needs it.
            if (pp.ValueKind == JsonValueKind.Object
                    && pp.TryGetProperty("single", out var single)
                    && single.ValueKind == JsonValueKind.Array)
            {
                var specials = single.EnumerateArray()
                    .Where(p => p.TryGetProperty("SpecialToken", out _))
                    .Select(p => p.GetProperty("SpecialToken").GetProperty("id").GetString()!)
                    .ToList();
                if (specials.Count >= 2)
                    (templateCls, templateSep) = (specials[0], specials[^1]);
            }
        }

        string clsStr, sepStr, padStr, unkStr, maskStr;
        if (File.Exists(configPath))
        {
            // 2a. Fused layout: tokenizer_config.json says "cls_token": "[CLS]" (for ModernBERT)
            //     or "cls_token": "<bos>" (for mmBERT). HuggingFace sometimes stores these as
            //     plain strings and sometimes as {"content": "...", ...} dicts.
            using var f = File.OpenRead(configPath);
            using var cfg = JsonDocument.Parse(f);
            var r = cfg.RootElement;
            clsStr  = RequireTokenString(r, "cls_token",  configPath);
            sepStr  = RequireTokenString(r, "sep_token",  configPath);
            padStr  = RequireTokenString(r, "pad_token",  configPath);
            unkStr  = RequireTokenString(r, "unk_token",  configPath);
            maskStr = RequireTokenString(r, "mask_token", configPath);
        }
        else
        {
            // 2b. Split layout: no tokenizer_config.json. CLS and SEP come from the
            //     post_processor template when it has one; a spelling guess is not enough there,
            //     because mmBERT's vocab holds both "<s>" and "<bos>" but only "<bos>" is its CLS.
            //     The other roles (and CLS/SEP without a template) are guessed from conventional
            //     special-token names — the ModernBERT bracketed form and the SentencePiece/BPE
            //     form — and accepted only if the spelling appears in tokenizer.json's token map.
            clsStr  = templateCls ?? FirstKnownToken(idByContent, "cls_token", configPath, "[CLS]", "<bos>", "<s>");
            sepStr  = templateSep ?? FirstKnownToken(idByContent, "sep_token", configPath, "[SEP]", "<eos>", "</s>");
            padStr  = FirstKnownToken(idByContent, "pad_token",  configPath, "[PAD]", "<pad>");
            maskStr = FirstKnownToken(idByContent, "mask_token", configPath, "[MASK]", "<mask>");
            unkStr  = FirstKnownToken(idByContent, "unk_token",  configPath, "[UNK]", "<unk>");
        }

        // 3. Resolve each role to its id. Any missing token here means the two files are
        //    from different checkpoints, which would silently produce wrong positions.
        return (
            Resolve(padStr,  "pad_token",  idByContent, configPath),
            Resolve(clsStr,  "cls_token",  idByContent, configPath),
            Resolve(sepStr,  "sep_token",  idByContent, configPath),
            Resolve(maskStr, "mask_token", idByContent, configPath),
            Resolve(unkStr,  "unk_token",  idByContent, configPath),
            maskStr
        );
    }

    /// <summary>
    /// Used only when <c>tokenizer_config.json</c> is absent (the split layout): returns the
    /// first of <paramref name="candidates"/> that is a known token in <paramref name="idByContent"/>,
    /// i.e. actually appears in <c>tokenizer.json</c>.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// When none of the candidate spellings is known — the loaded tokenizer.json uses a
    /// convention this method does not yet recognise for <paramref name="role"/>.
    /// </exception>
    private static string FirstKnownToken(
        Dictionary<string, int> idByContent, string role, string configPath, params string[] candidates)
    {
        foreach (var candidate in candidates)
            if (idByContent.ContainsKey(candidate))
                return candidate;

        throw new InvalidOperationException(
            $"Could not determine the '{role}' token: no 'tokenizer_config.json' was found "
            + $"next to the tokenizer (expected at '{configPath}'), and none of the conventional "
            + $"spellings ({string.Join(", ", candidates)}) appear in tokenizer.json's "
            + "added_tokens or post_processor.special_tokens.");
    }

    /// <summary>
    /// Extracts the token string for <paramref name="key"/> from a tokenizer_config root.
    /// HuggingFace can store a special token either as a plain string or as a
    /// <c>{"content": "...", ...}</c> dict; both are handled.
    /// </summary>
    private static string RequireTokenString(JsonElement root, string key, string configPath)
    {
        if (!root.TryGetProperty(key, out var val))
            throw new InvalidOperationException(
                $"'{configPath}' is missing the '{key}' key — cannot determine the special-token ids.");

        if (val.ValueKind == JsonValueKind.Object)
        {
            if (val.TryGetProperty("content", out var content))
                return content.GetString()
                    ?? throw new InvalidOperationException(
                        $"'{key}'.content is null in '{configPath}'.");
            throw new InvalidOperationException(
                $"'{key}' in '{configPath}' is a dict with no 'content' field.");
        }

        return val.GetString()
            ?? throw new InvalidOperationException($"'{key}' is null in '{configPath}'.");
    }

    /// <summary>
    /// Looks up <paramref name="token"/> in <paramref name="idByContent"/>, throwing a
    /// clear error when it is absent — which typically means the two tokenizer files are
    /// from different checkpoints and would silently produce wrong model outputs.
    /// </summary>
    private static int Resolve(string token, string role,
        Dictionary<string, int> idByContent, string configPath)
    {
        if (idByContent.TryGetValue(token, out var id)) return id;
        throw new InvalidOperationException(
            $"Special token '{token}' (role: {role}, declared in '{configPath}') was not found "
            + "in tokenizer.json's added_tokens or post_processor.special_tokens. "
            + "Check that tokenizer.json and tokenizer_config.json belong to the same checkpoint.");
    }
}
