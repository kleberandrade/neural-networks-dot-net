using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using NeuralNetworks.Internal;

namespace NeuralNetworks;

public sealed record TextTrainingState(int EpochsCompleted, int ShuffleSeed, double LearningRate, string CorpusFingerprint,
    int BatchSize = 1, string Optimizer = "sgd");
public sealed record LoadedTextCheckpoint(TextTransformer Model, CharacterTokenizer Tokenizer, TextTrainingState? Training, AdamOptimizer? Optimizer = null);
public sealed record TextCheckpointLimits(long MaxFileBytes = 32 * 1024 * 1024, int MaxParameters = 1000000,
    int MaxVocabularySize = 65536, int MaxSequenceLength = 4096, int MaxLayers = 64);

/// <summary>Versioned, checksummed JSON containing a text model and its exact character vocabulary.</summary>
public static class TextCheckpoint
{
    private const string Format = "neural-networks-dot-net.text";
    private const int Version = 3;
    private sealed record Configuration(int VocabularySize, int SourceLength, int TargetLength, int ModelSize,
        int HeadCount, int FeedForwardSize, int EncoderLayerCount, int DecoderLayerCount, int Seed);
    private sealed record Payload(Configuration Configuration, string[] Vocabulary, double[] Parameters, TextTrainingState? Training, AdamState? Optimizer);

    public static void Save(string path, TextTransformer model, CharacterTokenizer tokenizer, TextTrainingState? training = null, AdamOptimizer? optimizer = null)
    {
        ArgumentNullException.ThrowIfNull(model); ArgumentNullException.ThrowIfNull(tokenizer);
        if (model.VocabularySize != tokenizer.VocabularySize) throw new ArgumentException("Model and tokenizer vocabulary sizes differ.");
        ValidateState(training); optimizer?.EnsureModel(model); ValidateOptimizerPair(training, optimizer is not null);
        var config = new Configuration(model.VocabularySize, model.SourceLength, model.TargetLength, model.ModelSize,
            model.HeadCount, model.FeedForwardSize, model.EncoderLayerCount, model.DecoderLayerCount, model.InitializationSeed);
        var payload = JsonSerializer.SerializeToUtf8Bytes(new Payload(config, tokenizer.GetVocabulary(), model.GetParameters(), training, optimizer?.GetState()), TextFiles.Json);
        var hash = Convert.ToHexString(SHA256.HashData(payload)).ToLowerInvariant();
        using var output = new MemoryStream();
        using (var writer = new Utf8JsonWriter(output))
        {
            writer.WriteStartObject(); writer.WriteString("format", Format); writer.WriteNumber("version", Version);
            writer.WritePropertyName("payload"); writer.WriteRawValue(payload); writer.WriteString("sha256", hash); writer.WriteEndObject();
        }
        TextFiles.WriteAtomic(path, output.ToArray());
    }

    public static LoadedTextCheckpoint Load(string path, TextCheckpointLimits? limits = null)
    {
        var bound = limits ?? new TextCheckpointLimits(); ValidateLimits(bound);
        var bytes = TextFiles.Read(path, bound.MaxFileBytes);
        try
        {
            using var document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 32 });
            var root = document.RootElement; TextFiles.UniqueProperties(root); Names(root, "format", "version", "payload", "sha256");
            var version = root.GetProperty("version").GetInt32();
            if (root.GetProperty("format").GetString() != Format || (version != 1 && version != 2 && version != Version))
                throw new InvalidDataException("Unsupported text checkpoint format/version.");
            var payload = root.GetProperty("payload");
            if (version == 1) Names(payload, "configuration", "vocabulary", "parameters", "training");
            else Names(payload, "configuration", "vocabulary", "parameters", "training", "optimizer");
            var expectedHash = root.GetProperty("sha256").GetString();
            if (!IsHash(expectedHash) || !CryptographicOperations.FixedTimeEquals(Convert.FromHexString(expectedHash!),
                SHA256.HashData(Encoding.UTF8.GetBytes(payload.GetRawText()))))
                throw new InvalidDataException("Checkpoint checksum mismatch.");
            var config = payload.GetProperty("configuration").Deserialize<Configuration>(TextFiles.Json) ?? throw new InvalidDataException("Missing configuration.");
            var expected = ValidateConfiguration(config, bound);
            var parameters = payload.GetProperty("parameters"); var vocabulary = payload.GetProperty("vocabulary");
            if (parameters.GetArrayLength() != expected || vocabulary.GetArrayLength() != config.VocabularySize)
                throw new InvalidDataException("Checkpoint parameter/vocabulary dimensions differ from configuration.");
            var values = parameters.Deserialize<double[]>(TextFiles.Json)!;
            if (values.Any(x => !double.IsFinite(x))) throw new InvalidDataException("Checkpoint parameters must be finite.");
            var tokenizer = CharacterTokenizer.FromVocabulary(vocabulary.Deserialize<string[]>(TextFiles.Json)!);
            var progress = payload.GetProperty("training");
            if (version == 1 && progress.ValueKind != JsonValueKind.Null)
                Names(progress, "epochsCompleted", "shuffleSeed", "learningRate", "corpusFingerprint");
            if (version >= 2 && progress.ValueKind != JsonValueKind.Null)
                Names(progress, "epochsCompleted", "shuffleSeed", "learningRate", "corpusFingerprint", "batchSize", "optimizer");
            var training = progress.Deserialize<TextTrainingState?>(TextFiles.Json); ValidateState(training);
            var adamState = version == 1 ? null : ReadAdamState(payload.GetProperty("optimizer"), expected, version);
            ValidateOptimizerPair(training, adamState is not null);
            var model = new TextTransformer(config.VocabularySize, config.SourceLength, config.TargetLength, config.ModelSize,
                config.HeadCount, config.FeedForwardSize, config.EncoderLayerCount, config.DecoderLayerCount, config.Seed);
            model.SetParameters(values);
            return new LoadedTextCheckpoint(model, tokenizer, training, adamState is null ? null : AdamOptimizer.Restore(model, adamState));
        }
        catch (Exception error) when (error is JsonException or ArgumentException or InvalidOperationException or KeyNotFoundException or OverflowException or FormatException)
        { throw new InvalidDataException("Invalid text checkpoint: " + error.Message, error); }
    }

    private static int ValidateConfiguration(Configuration c, TextCheckpointLimits limits)
    {
        if (c.VocabularySize < 5 || c.VocabularySize > limits.MaxVocabularySize || c.SourceLength <= 0 || c.SourceLength > limits.MaxSequenceLength
            || c.TargetLength <= 0 || c.TargetLength > limits.MaxSequenceLength || c.ModelSize < 2 || c.HeadCount <= 0 || c.ModelSize % c.HeadCount != 0
            || c.FeedForwardSize <= 0 || c.EncoderLayerCount <= 0 || c.EncoderLayerCount > limits.MaxLayers || c.DecoderLayerCount <= 0 || c.DecoderLayerCount > limits.MaxLayers)
            throw new InvalidDataException("Checkpoint architecture is invalid or exceeds configured limits.");
        checked
        {
            long Linear(long input, long output) => output * (input + 1);
            var d = (long)c.ModelSize; var attention = 4 * Linear(d, d); var normalization = 2 * d;
            var ff = Linear(d, c.FeedForwardSize) + Linear(c.FeedForwardSize, d) + normalization;
            var encoder = attention + normalization + ff; var decoder = 2 * attention + 2 * normalization + ff;
            var count = 2 * d * c.VocabularySize + c.EncoderLayerCount * encoder + c.DecoderLayerCount * decoder + Linear(d, c.VocabularySize);
            if (count > limits.MaxParameters || count > int.MaxValue) throw new InvalidDataException("Checkpoint exceeds parameter limit.");
            return (int)count;
        }
    }

    private static void ValidateLimits(TextCheckpointLimits limits)
    {
        if (limits.MaxFileBytes <= 0 || limits.MaxFileBytes > int.MaxValue || limits.MaxParameters <= 0 || limits.MaxVocabularySize < 5
            || limits.MaxSequenceLength <= 0 || limits.MaxLayers <= 0) throw new ArgumentOutOfRangeException(nameof(limits));
    }

    private static bool IsHash(string? hash) => hash is { Length: 64 } && hash.All(Uri.IsHexDigit);
    private static void ValidateState(TextTrainingState? state)
    {
        if (state is not null && (state.EpochsCompleted < 0 || !double.IsFinite(state.LearningRate) || state.LearningRate <= 0 || !IsHash(state.CorpusFingerprint)
            || state.BatchSize <= 0 || state.Optimizer is not ("sgd" or "adam") || (state.Optimizer == "sgd" && state.BatchSize != 1)))
            throw new ArgumentException("Invalid training progress metadata.", nameof(state));
    }

    private static AdamState? ReadAdamState(JsonElement value, int parameterCount, int version)
    {
        if (value.ValueKind == JsonValueKind.Null) return null;
        Names(value, "options", "step", "firstMoment", "secondMoment");
        var options = value.GetProperty("options");
        if (version == 2) Names(options, "beta1", "beta2", "epsilon");
        else
        {
            Names(options, "beta1", "beta2", "epsilon", "maxGradientNorm", "schedule");
            var schedule = options.GetProperty("schedule");
            if (schedule.ValueKind != JsonValueKind.Null) Names(schedule, "warmupSteps", "decaySteps", "minimumFactor");
        }
        if (value.GetProperty("firstMoment").GetArrayLength() != parameterCount || value.GetProperty("secondMoment").GetArrayLength() != parameterCount)
            throw new InvalidDataException("Adam moments must match model parameter count.");
        var state = value.Deserialize<AdamState>(TextFiles.Json)!; AdamOptimizer.ValidateState(state, parameterCount);
        return state;
    }

    private static void ValidateOptimizerPair(TextTrainingState? training, bool hasAdam)
    {
        if (training is not null && (training.Optimizer == "adam") != hasAdam)
            throw new ArgumentException("Training optimizer and saved Adam state disagree.");
    }

    private static void Names(JsonElement value, params string[] names)
    {
        if (value.ValueKind != JsonValueKind.Object || !value.EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal).SequenceEqual(names.Order(StringComparer.Ordinal)))
            throw new InvalidDataException("Checkpoint object has missing or unknown fields.");
    }

}
