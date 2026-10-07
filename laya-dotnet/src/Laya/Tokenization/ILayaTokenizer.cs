namespace Laya.Tokenization;

/// <summary>
/// Tokenizes text for the Laya ONNX model, matching the behaviour of the Python
/// <c>tokenizers</c> library called with <c>add_special_tokens=False</c>.
/// </summary>
public interface ILayaTokenizer
{
    /// <summary>
    /// Encodes <paramref name="text"/> to token ids with <b>no</b> special tokens
    /// prepended or appended. Equivalent to Python's
    /// <c>tok(text, add_special_tokens=False).input_ids</c>.
    /// </summary>
    int[] Encode(string text);

    /// <summary>Token id of the padding token, used to right-pad sequences to a uniform length.</summary>
    int PadId { get; }

    /// <summary>
    /// Token id of the CLS token. Python prepends this to every sequence before the
    /// head fragment. Named <c>ClsId</c> to match the HuggingFace convention; the
    /// underlying token string varies by checkpoint (<c>&lt;bos&gt;</c> for mmBERT,
    /// <c>[CLS]</c> for ModernBERT).
    /// </summary>
    int ClsId { get; }

    /// <summary>
    /// Token id of the SEP token. Python inserts this between the head, the option
    /// fragments, and the state. Named <c>SepId</c> to match the HuggingFace convention;
    /// the underlying token string varies by checkpoint (<c>&lt;eos&gt;</c> for mmBERT,
    /// <c>[SEP]</c> for ModernBERT).
    /// </summary>
    int SepId { get; }

    /// <summary>
    /// Token id of the mask token. Each option fragment starts with one — its position
    /// in the sequence is recorded as a marker and later read by the scoring head.
    /// </summary>
    int MaskId { get; }

    /// <summary>Token id of the unknown-token.</summary>
    int UnkId { get; }

    /// <summary>
    /// The literal mask token string (e.g. <c>"&lt;mask&gt;"</c> for mmBERT or
    /// <c>"[MASK]"</c> for ModernBERT). Used to scrub any accidental occurrence of the
    /// mask token from user-supplied text before encoding.
    /// </summary>
    string MaskToken { get; }
}
