using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NeuralNetworks.Internal;

internal static class TextFiles
{
    public static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        RespectRequiredConstructorParameters = true,
        MaxDepth = 32
    };

    public static byte[] Read(string path, long maxBytes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (maxBytes <= 0 || maxBytes > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(maxBytes));
        using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
        if (input.Length > maxBytes) throw new InvalidDataException("File exceeds configured byte limit.");
        using var output = new MemoryStream(); var buffer = new byte[65536];
        int count;
        while ((count = input.Read(buffer)) > 0)
        {
            if (output.Length + count > maxBytes) throw new InvalidDataException("File exceeds configured byte limit.");
            output.Write(buffer, 0, count);
        }
        return output.ToArray();
    }

    public static string ReadUtf8(string path, long maxBytes)
    {
        try
        {
            var bytes = Read(path, maxBytes);
            var start = bytes.AsSpan().StartsWith(new byte[] { 0xef, 0xbb, 0xbf }) ? 3 : 0;
            return new UTF8Encoding(false, true).GetString(bytes.AsSpan(start));
        }
        catch (DecoderFallbackException error) { throw new InvalidDataException("Corpus must be valid UTF-8.", error); }
    }

    public static void UniqueProperties(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in element.EnumerateObject())
            {
                if (!names.Add(property.Name)) throw new InvalidDataException("Duplicate JSON property: " + property.Name);
                UniqueProperties(property.Value);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
            foreach (var item in element.EnumerateArray()) UniqueProperties(item);
    }

    public static void WriteAtomic(string path, byte[] bytes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var full = Path.GetFullPath(path); var directory = Path.GetDirectoryName(full)!;
        Directory.CreateDirectory(directory);
        var temporary = Path.Combine(directory, "." + Path.GetFileName(full) + "." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(bytes); stream.Flush(flushToDisk: true);
            }
            File.Move(temporary, full, overwrite: true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }
}
