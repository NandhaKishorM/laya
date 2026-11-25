using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// <see cref="LayaConfig.Parse"/> against Python's loader (<c>onnx_agent.py:192-262</c>) on configs whose fields have the
/// wrong JSON type, and against <c>build_sequence</c> for a wrong-typed <c>max_len</c> or
/// <c>head_max_len</c>. Expectations were produced by running the Python code on the same documents. One corner is
/// not ported and so not listed: a small float <c>head_max_len</c> that <c>build_head</c> survives (the typical float fails).
/// </summary>
public sealed class ConfigFieldTypeTests
{
    public static TheoryData<string, string?, double[], string[]> Loads() => new()
    {
        { "{\"temperature\": 5}", "TypeError: 'int' object is not iterable", new double[0], new string[0] },
        { "{\"temperature\": 1.5}", "TypeError: 'float' object is not iterable", new double[0], new string[0] },
        { "{\"temperature\": true}", "TypeError: 'bool' object is not iterable", new double[0], new string[0] },
        { "{\"temperature\": null}", "TypeError: 'NoneType' object is not iterable", new double[0], new string[0] },
        { "{\"temperature\": \"abc\"}", null, new double[] { 1.0, 1.0, 1.0 }, new string[] { "temperature[0]='a' -> 1", "temperature[1]='b' -> 1", "temperature[2]='c' -> 1" } },
        { "{\"temperature\": \"\"}", null, new double[] {  }, new string[] {  } },
        { "{\"temperature\": {\"a\": 1, \"b\": 2}}", null, new double[] { 1.0, 1.0 }, new string[] { "temperature[0]='a' -> 1", "temperature[1]='b' -> 1" } },
        { "{\"temperature\": []}", null, new double[] {  }, new string[] {  } },
        { "{\"temperature\": [1]}", null, new double[] { 1.0 }, new string[] {  } },
        { "{\"temperature\": [1, 2, 3]}", null, new double[] { 1.0, 2.0, 3.0 }, new string[] {  } },
        { "{\"temperature\": \"a\ud835\udc00b\"}", null, new double[] { 1.0, 1.0, 1.0 }, new string[] { "temperature[0]='a' -> 1", "temperature[1]='\ud835\udc00' -> 1", "temperature[2]='b' -> 1" } },
        { "{\"temperature\": 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000}", "TypeError: 'int' object is not iterable", new double[0], new string[0] },
        { "{\"temperature\": [0.2, \"x\", true]}", null, new double[] { 0.5, 1.0, 1.0 }, new string[] { "temperature[0]=0.2 -> 0.5", "temperature[1]='x' -> 1", "temperature[2]=True -> 1" } },
        { "{\"temperature_by_options\": [1]}", "AttributeError: 'list' object has no attribute 'items'", new double[0], new string[0] },
        { "{\"temperature_by_options\": \"x\"}", "AttributeError: 'str' object has no attribute 'items'", new double[0], new string[0] },
        { "{\"temperature_by_options\": 5}", "AttributeError: 'int' object has no attribute 'items'", new double[0], new string[0] },
        { "{\"temperature_by_options\": null}", "AttributeError: 'NoneType' object has no attribute 'items'", new double[0], new string[0] },
        { "{\"temperature_by_options\": true}", "AttributeError: 'bool' object has no attribute 'items'", new double[0], new string[0] },
        { "{\"temperature_by_options\": {}}", null, new double[] { 1.0, 1.0, 1.0 }, new string[] {  } },
        { "{\"temperature_by_options\": {\"a\": 0.1}}", null, new double[] { 1.0, 1.0, 1.0 }, new string[] { "a=0.1 -> 0.5" } },
        { "{\"temperature_by_options\": {\"a\": 1, \"b\": \"q\"}}", null, new double[] { 1.0, 1.0, 1.0 }, new string[] { "b='q' -> 1" } },
        { "{\"temperature_by_options\": 1.5}", "AttributeError: 'float' object has no attribute 'items'", new double[0], new string[0] },
        { "{\"temperature\": 5, \"temperature_by_options\": [1]}", "TypeError: 'int' object is not iterable", new double[0], new string[0] },
        { "{\"temperature\": 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000, \"temperature_by_options\": 5}", "TypeError: 'int' object is not iterable", new double[0], new string[0] },
        { "{\"temperature\": [1, 1, 1], \"temperature_by_options\": {\"a\": 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000}}", null, new double[] { 1.0, 1.0, 1.0 }, new string[] { "a=1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 -> 1" } },
        { "{}", null, new double[] { 1.0, 1.0, 1.0 }, new string[] {  } },
        { "{\"temperature\": [0.2, 1, 1], \"temperature_by_options\": {\"x\": 9}}", null, new double[] { 0.5, 1.0, 1.0 }, new string[] { "x=9 -> 5", "temperature[0]=0.2 -> 0.5" } },
        { "{\"option_layout\": \"sequential\"}", null, new double[] { 1.0, 1.0, 1.0 }, new string[] {  } },
        { "{\"option_layout\": \"parallel\"}", "ValueError: option_layout='parallel' is not supported by the ONNX runtime; load this checkpoint with laya.Agent", new double[0], new string[0] },
        { "{\"option_layout\": null}", "ValueError: option_layout=None is not supported by the ONNX runtime; load this checkpoint with laya.Agent", new double[0], new string[0] },
        { "{\"option_layout\": 5}", "ValueError: option_layout=5 is not supported by the ONNX runtime; load this checkpoint with laya.Agent", new double[0], new string[0] },
        { "{\"option_layout\": true}", "ValueError: option_layout=True is not supported by the ONNX runtime; load this checkpoint with laya.Agent", new double[0], new string[0] },
        { "{\"option_layout\": [\"a\"]}", "ValueError: option_layout=['a'] is not supported by the ONNX runtime; load this checkpoint with laya.Agent", new double[0], new string[0] },
        { "{\"option_layout\": {\"k\": 1}}", "ValueError: option_layout={'k': 1} is not supported by the ONNX runtime; load this checkpoint with laya.Agent", new double[0], new string[0] },
        { "{\"option_layout\": \"x\ue000\"}", "ValueError: option_layout='x\\ue000' is not supported by the ONNX runtime; load this checkpoint with laya.Agent", new double[0], new string[0] },
        { "[1]", "AttributeError: 'list' object has no attribute 'get'", new double[0], new string[0] },
        { "\"x\"", "AttributeError: 'str' object has no attribute 'get'", new double[0], new string[0] },
        { "5", "AttributeError: 'int' object has no attribute 'get'", new double[0], new string[0] },
        { "null", "AttributeError: 'NoneType' object has no attribute 'get'", new double[0], new string[0] },
        { "true", "AttributeError: 'bool' object has no attribute 'get'", new double[0], new string[0] },
        { "1.5", "AttributeError: 'float' object has no attribute 'get'", new double[0], new string[0] },
    };

    [Theory]
    [MemberData(nameof(Loads))]
    public void LoadMatchesPython(string json, string? error, double[] temperatures, string[] clamped)
    {
        if (error is not null)
        {
            var thrown = Assert.ThrowsAny<Exception>(() => LayaConfig.Parse(json));
            Assert.True(thrown is InvalidDataException or JsonException, thrown.GetType().Name);
            Assert.Equal(error, thrown.Message);
            return;
        }
        var config = LayaConfig.Parse(json);
        // .NET pads a short list to the three question types with 1.0; Python would fail to index it at prediction.
        Assert.Equal<IEnumerable<double>>(temperatures, config.Temperature.Take(temperatures.Length));
        Assert.Equal<IEnumerable<string>>(clamped, config.ClampedTemperatures);
    }

    public static TheoryData<string, string?> Lengths() => new()
    {
        { "{}", null },
        { "{\"head_max_len\": 512}", null },
        { "{\"head_max_len\": true}", null },
        { "{\"head_max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"head_max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"head_max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"head_max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": 512}", null },
        { "{\"max_len\": 512, \"head_max_len\": 512}", null },
        { "{\"max_len\": 512, \"head_max_len\": true}", null },
        { "{\"max_len\": 512, \"head_max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": 512, \"head_max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": 512, \"head_max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": 512, \"head_max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": true}", null },
        { "{\"max_len\": true, \"head_max_len\": 512}", null },
        { "{\"max_len\": true, \"head_max_len\": true}", null },
        { "{\"max_len\": true, \"head_max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": true, \"head_max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": true, \"head_max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": true, \"head_max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": 1.5}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": 1.5, \"head_max_len\": 512}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": 1.5, \"head_max_len\": true}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": 1.5, \"head_max_len\": 1.5}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": 1.5, \"head_max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": 1.5, \"head_max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": 1.5, \"head_max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": 1.5, \"head_max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": \"x\", \"head_max_len\": 512}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": \"x\", \"head_max_len\": true}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": \"x\", \"head_max_len\": 1.5}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": \"x\", \"head_max_len\": 192.0}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": \"x\", \"head_max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": \"x\", \"head_max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": \"x\", \"head_max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": \"x\", \"head_max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": null, \"head_max_len\": 512}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": null, \"head_max_len\": true}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": null, \"head_max_len\": 1.5}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": null, \"head_max_len\": 192.0}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": null, \"head_max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": null, \"head_max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": null, \"head_max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": null, \"head_max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": [1], \"head_max_len\": 512}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": [1], \"head_max_len\": true}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": [1], \"head_max_len\": 1.5}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": [1], \"head_max_len\": 192.0}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": [1], \"head_max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": [1], \"head_max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": [1], \"head_max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": [1], \"head_max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": {}, \"head_max_len\": 512}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": {}, \"head_max_len\": true}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
        { "{\"max_len\": {}, \"head_max_len\": 1.5}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": {}, \"head_max_len\": 192.0}", "TypeError: slice indices must be integers or None or have an __index__ method" },
        { "{\"max_len\": {}, \"head_max_len\": \"x\"}", "TypeError: unsupported operand type(s) for -: 'str' and 'int'" },
        { "{\"max_len\": {}, \"head_max_len\": null}", "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'" },
        { "{\"max_len\": {}, \"head_max_len\": [1]}", "TypeError: unsupported operand type(s) for -: 'list' and 'int'" },
        { "{\"max_len\": {}, \"head_max_len\": {}}", "TypeError: unsupported operand type(s) for -: 'dict' and 'int'" },
    };

    [Theory]
    [MemberData(nameof(Lengths))]
    public void WrongTypedLengthsAreRefusedAtLoadWithPythonsTypeError(string json, string? error)
    {
        if (error is null) { LayaConfig.Parse(json); return; }
        Assert.Equal(error, Assert.Throws<InvalidDataException>(() => LayaConfig.Parse(json)).Message);
    }

    [Theory]
    [InlineData("{\"max_len\": -1}", "max_len=-1: must be a non-negative integer")]
    [InlineData("{\"head_max_len\": -192}", "head_max_len=-192: must be a non-negative integer")]
    [InlineData("{\"max_len\": -99999999999999999999999}", "max_len=-99999999999999999999999: must be a non-negative integer")]
    public void NegativeLengthsAreRefusedAtLoad(string json, string error) =>
        Assert.Equal(error, Assert.Throws<InvalidDataException>(() => LayaConfig.Parse(json)).Message);

    [Theory]
    [InlineData("{\"max_len\": 0, \"head_max_len\": 0}", 0, 0)]
    [InlineData("{\"max_len\": true, \"head_max_len\": false}", 1, 0)]
    [InlineData("{\"max_len\": 1024, \"head_max_len\": 256}", 1024, 256)]
    [InlineData("{}", 512, 192)]
    [InlineData("{\"max_len\": 99999999999999999999}", int.MaxValue, 192)]
    public void NonNegativeIntsAndBoolsAreAccepted(string json, int maxLen, int headMaxLen)
    {
        var config = LayaConfig.Parse(json);
        Assert.Equal((maxLen, headMaxLen), (config.MaxLen, config.HeadMaxLen));
    }

    // Python evaluates `temperature[qt]` as the default of the by-options lookup, so a short list fails for that question
    // type even when its bucket is present. Two option buckets (k = 2) are used throughout.
    [Theory]
    [InlineData("{\"temperature\": [], \"temperature_by_options\": {}}", 0, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [], \"temperature_by_options\": {}}", 1, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [], \"temperature_by_options\": {}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 0, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 1, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [2.0], \"temperature_by_options\": {}}", 0, "2.0")]
    [InlineData("{\"temperature\": [2.0], \"temperature_by_options\": {}}", 1, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [2.0], \"temperature_by_options\": {}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [2.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 0, "0.7")]
    [InlineData("{\"temperature\": [2.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 1, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [2.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [2.0, 3.0], \"temperature_by_options\": {}}", 0, "2.0")]
    [InlineData("{\"temperature\": [2.0, 3.0], \"temperature_by_options\": {}}", 1, "3.0")]
    [InlineData("{\"temperature\": [2.0, 3.0], \"temperature_by_options\": {}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [2.0, 3.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 0, "0.7")]
    [InlineData("{\"temperature\": [2.0, 3.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 1, "0.9")]
    [InlineData("{\"temperature\": [2.0, 3.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": [2.0, 3.0, 4.0], \"temperature_by_options\": {}}", 0, "2.0")]
    [InlineData("{\"temperature\": [2.0, 3.0, 4.0], \"temperature_by_options\": {}}", 1, "3.0")]
    [InlineData("{\"temperature\": [2.0, 3.0, 4.0], \"temperature_by_options\": {}}", 2, "4.0")]
    [InlineData("{\"temperature\": [2.0, 3.0, 4.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 0, "0.7")]
    [InlineData("{\"temperature\": [2.0, 3.0, 4.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 1, "0.9")]
    [InlineData("{\"temperature\": [2.0, 3.0, 4.0], \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 2, "1.3")]
    [InlineData("{\"temperature\": \"ab\", \"temperature_by_options\": {}}", 0, "1.0")]
    [InlineData("{\"temperature\": \"ab\", \"temperature_by_options\": {}}", 1, "1.0")]
    [InlineData("{\"temperature\": \"ab\", \"temperature_by_options\": {}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": \"ab\", \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 0, "0.7")]
    [InlineData("{\"temperature\": \"ab\", \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 1, "0.9")]
    [InlineData("{\"temperature\": \"ab\", \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": {\"p\": 1}, \"temperature_by_options\": {}}", 0, "1.0")]
    [InlineData("{\"temperature\": {\"p\": 1}, \"temperature_by_options\": {}}", 1, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": {\"p\": 1}, \"temperature_by_options\": {}}", 2, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": {\"p\": 1}, \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 0, "0.7")]
    [InlineData("{\"temperature\": {\"p\": 1}, \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 1, "IndexError: list index out of range")]
    [InlineData("{\"temperature\": {\"p\": 1}, \"temperature_by_options\": {\"choice:2\": 0.7, \"score:2\": 0.9, \"noul:2\": 1.3}}", 2, "IndexError: list index out of range")]
    public void AShortTemperatureListFailsLikePythonWhenAQuestionNeedsIt(string json, int type, string expected)
    {
        var config = LayaConfig.Parse(json);
        try
        {
            var value = Calibration.ResolveTemperature(config, (QuestionType)type, 2);
            Assert.Equal(expected, PythonFloat.Repr(value));
        }
        catch (InvalidOperationException e)
        {
            Assert.Equal(expected, e.Message);
        }
    }
}
