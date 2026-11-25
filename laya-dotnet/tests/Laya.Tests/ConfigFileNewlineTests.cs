using System.Text;
using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Config files are read in text mode by Python, so a CRLF or a lone CR reads as LF before <c>json.load</c> counts
/// lines. The expected messages were produced by Python reading the same bytes from a file.
/// </summary>
public sealed class ConfigFileNewlineTests
{
    public static TheoryData<string, string?> Cases() => new()
    {
        { "{\u000d\u000a  \"a\": 1,\u000d\u000a  x\u000d\u000a}", "Expecting property name enclosed in double quotes: line 3 column 3 (char 14)" },
        { "{\u000d  \"a\": 1,\u000d  x\u000d}", "Expecting property name enclosed in double quotes: line 3 column 3 (char 14)" },
        { "{\u000d\u000a\"a\": 1\u000d\u000a\"b\": 2}", "Expecting ',' delimiter: line 3 column 1 (char 9)" },
        { "{\"a\": 1}\u000d\u000a\u000d\u000agarbage", "Extra data: line 3 column 1 (char 10)" },
        { "{\u000d\u000a  \"a\": \"line\u000d\u000abreak\"\u000d\u000a}", "Invalid control character at: line 2 column 13 (char 14)" },
        { "{\"a\":\u000d\u000a\u000d\u000a}", "Expecting value: line 3 column 1 (char 7)" },
        { "{\"max_len\": 64,\u000d\"temperature\": [1,]}", "Expecting value: line 2 column 19 (char 34)" },
        { "{\"max_len\": 64}\u000d\u000a", null },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void FileErrorsCountLinesAsTextModeDoes(string content, string? message)
    {
        var path = Path.Combine(Path.GetTempPath(), "laya-nl-" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllBytes(path, Encoding.UTF8.GetBytes(content));
        try
        {
            if (message is null) { LayaConfig.Load(path); return; }
            Assert.Equal(message, Assert.Throws<JsonException>(() => LayaConfig.Load(path)).Message);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void TokenizerConfigIsReadTheSameWay()
    {
        var text = PythonJsonReader.ReadAllText(WriteTemp("{\"a\":\r\n1,\r}"));
        Assert.Equal("{\"a\":\n1,\n}", text);
    }

    private static string WriteTemp(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), "laya-nl-" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllBytes(path, Encoding.UTF8.GetBytes(content));
        return path;
    }
}
