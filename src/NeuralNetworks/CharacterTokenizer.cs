using System.Buffers;
using System.Text;

namespace NeuralNetworks;

/// <summary>An immutable, corpus-fitted Unicode scalar tokenizer; no normalization or subword merging.</summary>
public sealed class CharacterTokenizer
{
    public const int PadId = 0;
    public const int BosId = 1;
    public const int EosId = 2;
    public const int UnkId = 3;
    private static readonly string[] SpecialVocabulary = ["<pad>", "<bos>", "<eos>", "<unk>"];
    private readonly string[] vocabulary;
    private readonly Dictionary<int, int> ids;
    public int VocabularySize => vocabulary.Length;

    private CharacterTokenizer(int[] codePoints)
    {
        vocabulary = [.. SpecialVocabulary, .. codePoints.Select(x => new Rune(x).ToString())];
        ids = codePoints.Select((point, index) => (point, index)).ToDictionary(x => x.point, x => x.index + 4);
    }

    public static CharacterTokenizer FromCorpus(IEnumerable<string> corpus)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        var codePoints = new SortedSet<int>();
        foreach (var text in corpus) codePoints.UnionWith(ReadScalars(text));
        if (codePoints.Count == 0) throw new ArgumentException("Corpus must contain at least one Unicode scalar.", nameof(corpus));
        return new CharacterTokenizer(codePoints.ToArray());
    }

    public static CharacterTokenizer FromVocabulary(string[] vocabulary)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        if (vocabulary.Length < 5 || !vocabulary.Take(4).SequenceEqual(SpecialVocabulary))
            throw new ArgumentException("Vocabulary has invalid reserved IDs.", nameof(vocabulary));
        var points = new int[vocabulary.Length - 4];
        for (var i = 4; i < vocabulary.Length; i++)
        {
            var scalar = ReadScalars(vocabulary[i]);
            if (scalar.Length != 1 || (i > 4 && scalar[0] <= points[i - 5]))
                throw new ArgumentException("Vocabulary must contain unique, ascending Unicode scalars.", nameof(vocabulary));
            points[i - 4] = scalar[0];
        }
        return new CharacterTokenizer(points);
    }

    public string[] GetVocabulary() => (string[])vocabulary.Clone();

    public int[] Encode(string text, bool addBos = false, bool addEos = false)
    {
        var result = new List<int>();
        if (addBos) result.Add(BosId);
        foreach (var point in ReadScalars(text)) result.Add(ids.GetValueOrDefault(point, UnkId));
        if (addEos) result.Add(EosId);
        return result.ToArray();
    }

    public string Decode(int[] tokens, bool stopAtEos = true)
    {
        TokenSequences.ValidateIds(tokens, VocabularySize);
        var result = new StringBuilder();
        foreach (var token in tokens)
        {
            if (token == EosId && stopAtEos) break;
            if (token == UnkId) result.Append('�');
            else if (token >= 4) result.Append(vocabulary[token]);
        }
        return result.ToString();
    }

    internal static int[] ReadScalars(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var result = new List<int>();
        for (var offset = 0; offset < text.Length;)
        {
            var status = Rune.DecodeFromUtf16(text.AsSpan(offset), out var rune, out var consumed);
            if (status != OperationStatus.Done) throw new ArgumentException("Text contains malformed UTF-16.", nameof(text));
            result.Add(rune.Value); offset += consumed;
        }
        return result.ToArray();
    }
}

internal static class TokenSequences
{
    public static void ValidateIds(int[] tokens, int vocabularySize)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Any(id => id < 0 || id >= vocabularySize)) throw new ArgumentOutOfRangeException(nameof(tokens), "Token ID is outside the vocabulary.");
    }

    public static int[] Prefix(int[] tokens, int capacity, int vocabularySize)
    {
        ValidateIds(tokens, vocabularySize);
        if (tokens.Length == 0 || tokens.Length > capacity) throw new ArgumentException("Token sequence must fit the nonempty configured capacity.", nameof(tokens));
        var valid = Array.IndexOf(tokens, CharacterTokenizer.PadId);
        if (valid < 0) valid = tokens.Length;
        if (valid == 0 || tokens.Skip(valid).Any(id => id != CharacterTokenizer.PadId))
            throw new ArgumentException("Provide a nonempty valid prefix followed only by PAD.", nameof(tokens));
        return tokens.AsSpan(0, valid).ToArray();
    }
}
