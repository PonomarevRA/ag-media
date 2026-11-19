using System.Text.Json;
using AgPrivacy;

if (args.Length != 1) return 2;

try
{
    // Load the original weights once per batch. Standard output contains positions only.
    using var weights = File.OpenRead(args[0]);
    if (weights.Length > MicroPersonModel.MaximumFileBytes) return 2;
    var bytes = new byte[(int)weights.Length];
    weights.ReadExactly(bytes);
    var model = MicroPersonModel.Load(bytes);

    var texts = JsonSerializer.Deserialize<string[]>(Console.In.ReadToEnd());
    if (texts is null) return 2;
    var rows = texts.Select(text => FindMentions(model, text));
    Console.WriteLine(JsonSerializer.Serialize(new { version = model.Version, rows }));
    return 0;
}
catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException or ArgumentException)
{
    return 2;
}

/// <summary>Return UTF-16 positions, preserving the same indexing used by browser strings.</summary>
static IEnumerable<object> FindMentions(MicroPersonModel model, string text)
{
    IReadOnlyList<EntitySpan> spans;
    try { spans = model.Find(text); }
    catch (Exception e) when (e is ArgumentException or InvalidOperationException) { return []; }
    return spans.Select(span => new { start = span.Start, end = span.Start + span.Length });
}
