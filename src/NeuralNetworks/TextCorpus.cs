using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using NeuralNetworks.Internal;

namespace NeuralNetworks;

public sealed record TextPair(string Prompt, string Response);
public sealed record EncodedTextPair(int[] Source, int[] Target);
public sealed record TextCorpusLimits(long MaxBytes = 8 * 1024 * 1024, int MaxSamples = 10000, int MaxScalarsPerField = 4095);

/// <summary>A bounded in-memory teaching corpus loaded from UTF-8 pairs or continuation windows.</summary>
public sealed class TextCorpus
{
    public IReadOnlyList<TextPair> Pairs { get; }
    public string Fingerprint { get; }
    private TextCorpus(List<TextPair> pairs)
    {
        Pairs = Array.AsReadOnly(pairs.ToArray());
        Fingerprint = Convert.ToHexString(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(Pairs, TextFiles.Json))).ToLowerInvariant();
    }

    public static TextCorpus LoadPairs(string path, TextCorpusLimits? limits = null)
    {
        var bound = ValidateLimits(limits); var text = TextFiles.ReadUtf8(path, bound.MaxBytes);
        var pairs = new List<TextPair>(); using var reader = new StringReader(text);
        string? line; var number = 0;
        while ((line = reader.ReadLine()) is not null)
        {
            number++;
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                using var document = JsonDocument.Parse(line, new JsonDocumentOptions { MaxDepth = 32 });
                TextFiles.UniqueProperties(document.RootElement);
                var pair = document.RootElement.Deserialize<TextPair>(TextFiles.Json) ?? throw new JsonException("Expected an object.");
                Add(pairs, pair, bound);
            }
            catch (Exception error) when (error is JsonException or ArgumentException)
            { throw new InvalidDataException($"Invalid corpus record at line {number}: {error.Message}", error); }
        }
        return Finish(pairs);
    }

    public static TextCorpus LoadText(string path, int promptLength, int responseLength, int stride = 1, TextCorpusLimits? limits = null)
    {
        Guard.Positive(promptLength, nameof(promptLength)); Guard.Positive(responseLength, nameof(responseLength)); Guard.Positive(stride, nameof(stride));
        var width = checked(promptLength + responseLength); var bound = ValidateLimits(limits);
        if (promptLength > bound.MaxScalarsPerField || responseLength > bound.MaxScalarsPerField) throw new ArgumentOutOfRangeException(nameof(promptLength));
        var text = TextFiles.ReadUtf8(path, bound.MaxBytes).ReplaceLineEndings("\n");
        var points = CharacterTokenizer.ReadScalars(text); var pairs = new List<TextPair>();
        for (long start = 0; start + width <= points.Length; start += stride)
        {
            string Segment(int offset, int length) => string.Concat(points.AsSpan(offset, length).ToArray().Select(x => new Rune(x).ToString()));
            Add(pairs, new TextPair(Segment((int)start, promptLength), Segment((int)start + promptLength, responseLength)), bound);
        }
        return Finish(pairs);
    }

    public CharacterTokenizer CreateTokenizer() => CharacterTokenizer.FromCorpus(Pairs.SelectMany(p => new[] { p.Prompt, p.Response }));

    public EncodedTextPair[] Encode(CharacterTokenizer tokenizer, int? sourceLength = null, int? targetLength = null, bool allowUnknown = false)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (sourceLength is <= 0 || targetLength is <= 0) throw new ArgumentOutOfRangeException(nameof(sourceLength));
        return Pairs.Select(pair =>
        {
            var source = tokenizer.Encode(pair.Prompt, addEos: true); var target = tokenizer.Encode(pair.Response, addEos: true);
            if ((!allowUnknown && (source.Contains(CharacterTokenizer.UnkId) || target.Contains(CharacterTokenizer.UnkId)))
                || source.Length > (sourceLength ?? int.MaxValue) || target.Length > (targetLength ?? int.MaxValue))
                throw new ArgumentException("Corpus contains unknown characters or exceeds model token capacities.");
            return new EncodedTextPair(source, target);
        }).ToArray();
    }

    private static TextCorpusLimits ValidateLimits(TextCorpusLimits? limits)
    {
        var bound = limits ?? new TextCorpusLimits();
        if (bound.MaxBytes <= 0 || bound.MaxBytes > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(limits));
        Guard.Positive(bound.MaxSamples, nameof(limits)); Guard.Positive(bound.MaxScalarsPerField, nameof(limits));
        return bound;
    }

    private static void Add(List<TextPair> pairs, TextPair pair, TextCorpusLimits limits)
    {
        var a = CharacterTokenizer.ReadScalars(pair.Prompt); var b = CharacterTokenizer.ReadScalars(pair.Response);
        if (a.Length + b.Length == 0 || a.Length > limits.MaxScalarsPerField || b.Length > limits.MaxScalarsPerField)
            throw new InvalidDataException("Corpus fields are empty or exceed the scalar limit.");
        if (pairs.Count >= limits.MaxSamples) throw new InvalidDataException("Corpus exceeds the sample limit.");
        pairs.Add(pair);
    }

    private static TextCorpus Finish(List<TextPair> pairs) => pairs.Count == 0
        ? throw new InvalidDataException("Corpus contains no complete training samples.") : new TextCorpus(pairs);
}
