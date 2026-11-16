using NeuralNetworks.Internal;

namespace NeuralNetworks;

public sealed record TokenGeneration(int[] Tokens, bool EndedWithEos);

/// <summary>A small character/token encoder-decoder trained with mean categorical cross-entropy.</summary>
public sealed class TextTransformer : ParameterizedModel
{
    private readonly EmbeddingBlock sourceEmbedding;
    private readonly EmbeddingBlock targetEmbedding;
    private readonly TransformerEncoderLayer[] encoders;
    private readonly TransformerDecoderLayer[] decoders;
    private readonly LinearBlock output;
    public int VocabularySize { get; }
    public int SourceLength { get; }
    public int TargetLength { get; }
    public int ModelSize { get; }
    public int HeadCount { get; }
    public int FeedForwardSize { get; }
    public int InitializationSeed { get; }
    public int EncoderLayerCount { get; }
    public int DecoderLayerCount { get; }

    public TextTransformer(int vocabularySize, int sourceLength, int targetLength, int modelSize = 8, int headCount = 2,
        int feedForwardSize = 16, int encoderLayerCount = 2, int decoderLayerCount = 2, int seed = 0) : base(seed)
    {
        if (vocabularySize < 5) throw new ArgumentOutOfRangeException(nameof(vocabularySize), "Include four special tokens and at least one character.");
        VocabularySize = vocabularySize;
        SourceLength = Guard.Positive(sourceLength, nameof(sourceLength));
        TargetLength = Guard.Positive(targetLength, nameof(targetLength));
        MultiHeadAttention.Validate(modelSize, headCount);
        Guard.Positive(feedForwardSize, nameof(feedForwardSize));
        ModelSize = modelSize; HeadCount = headCount;
        FeedForwardSize = feedForwardSize; InitializationSeed = seed;
        EncoderLayerCount = Guard.Positive(encoderLayerCount, nameof(encoderLayerCount));
        DecoderLayerCount = Guard.Positive(decoderLayerCount, nameof(decoderLayerCount));
        sourceEmbedding = AddEmbedding(vocabularySize, modelSize);
        targetEmbedding = AddEmbedding(vocabularySize, modelSize);
        var builder = new TransformerBuilder(AddLinear, AddNormalization, modelSize, headCount, feedForwardSize);
        encoders = Enumerable.Range(0, EncoderLayerCount).Select(_ => builder.Encoder()).ToArray();
        decoders = Enumerable.Range(0, DecoderLayerCount).Select(_ => builder.Decoder()).ToArray();
        output = AddLinear(modelSize, vocabularySize);
    }

    /// <summary>Returns one vocabulary probability row per valid decoder-prefix position.</summary>
    public double[][] Predict(int[] source, int[] decoderInput)
    {
        var sample = PreparePrefix(source, decoderInput);
        return Evaluate(sample.Source, sample.Decoder).Logits.Select(Categorical.Probabilities).ToArray();
    }

    /// <summary>Target must end with EOS; the decoder input is shifted internally.</summary>
    public double Loss(int[] source, int[] target)
    {
        var sample = PrepareTarget(source, target);
        var graph = Evaluate(sample.Source, sample.Decoder);
        return Objective(graph.Logits, sample.Target).Value;
    }

    public double[] ComputeGradient(int[] source, int[] target)
    {
        return LossAndGradient(source, target).Gradient;
    }

    internal (double Loss, double[] Gradient) LossAndGradient(int[] source, int[] target)
    {
        var sample = PrepareTarget(source, target);
        var graph = Evaluate(sample.Source, sample.Decoder);
        var loss = Objective(graph.Logits, sample.Target);
        return (loss.Value, Derivatives(loss, graph.Parameters));
    }

    public double TrainSample(int[] source, int[] target, double learningRate)
    {
        Guard.LearningRate(learningRate);
        var sample = PrepareTarget(source, target);
        var graph = Evaluate(sample.Source, sample.Decoder);
        var loss = Objective(graph.Logits, sample.Target);
        ApplyUpdate(Derivatives(loss, graph.Parameters), learningRate);
        return loss.Value;
    }

    public TransformerStackAttentionWeights GetAttentionWeights(int[] source, int[] decoderInput)
    {
        var sample = PreparePrefix(source, decoderInput);
        var graph = CreateParameterGraph();
        var stack = Build(sample.Source, sample.Decoder, graph.Parameters);
        return new TransformerStackAttentionWeights(TransformerStacks.Snapshot(stack.Encoder),
            TransformerStacks.Snapshot(stack.Decoder), TransformerStacks.Snapshot(stack.Cross));
    }

    /// <summary>Greedy IDs excluding PAD/BOS; includes EOS when emitted and reports whether EOS was reached.</summary>
    public TokenGeneration Generate(int[] source, int? maxNewTokens = null) =>
        Decode(source, maxNewTokens, TokenSampling.Greedy);

    public TokenGeneration Sample(int[] source, TextSamplingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        TokenSampling.Validate(options, VocabularySize);
        var random = new Random(options.Seed);
        return Decode(source, options.MaxNewTokens, logits => TokenSampling.Choose(logits, options, random));
    }

    private TokenGeneration Decode(int[] source, int? maxNewTokens, Func<Scalar[], int> choose)
    {
        var sourceIds = TokenSequences.Prefix(source, SourceLength, VocabularySize);
        var limit = maxNewTokens ?? TargetLength;
        if (limit <= 0 || limit > TargetLength) throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
        var decoder = new List<int> { CharacterTokenizer.BosId }; var generated = new List<int>();
        for (var i = 0; i < limit; i++)
        {
            var logits = Evaluate(sourceIds, decoder.ToArray()).Logits[^1]; var next = choose(logits);
            generated.Add(next);
            if (next == CharacterTokenizer.EosId) return new TokenGeneration(generated.ToArray(), true);
            decoder.Add(next);
        }
        return new TokenGeneration(generated.ToArray(), false);
    }

    private StackResult Build(int[] source, int[] decoder, Scalar[] parameters) =>
        TransformerStacks.Apply(sourceEmbedding.Apply(source, parameters), targetEmbedding.Apply(decoder, parameters),
            parameters, encoders, decoders);

    private (Scalar[][] Logits, Scalar[] Parameters) Evaluate(int[] source, int[] decoder)
    {
        var graph = CreateParameterGraph();
        var decoded = Build(source, decoder, graph.Parameters).Output;
        return (decoded.Select(t => output.Apply(graph.Parameters, t)).ToArray(), graph.Parameters);
    }

    private static Scalar Objective(Scalar[][] logits, int[] target)
    {
        var loss = logits[0][0].Constant(0);
        for (var i = 0; i < target.Length; i++) loss += Categorical.CrossEntropy(logits[i], target[i]) / target.Length;
        return loss;
    }

    private (int[] Source, int[] Decoder) PreparePrefix(int[] source, int[] decoder)
    {
        var s = TokenSequences.Prefix(source, SourceLength, VocabularySize);
        var d = TokenSequences.Prefix(decoder, TargetLength, VocabularySize);
        if (d[0] != CharacterTokenizer.BosId || d.Skip(1).Any(id => id == CharacterTokenizer.BosId))
            throw new ArgumentException("Decoder prefix must begin with exactly one BOS.", nameof(decoder));
        return (s, d);
    }

    internal EncodedTextPair CopyTrainingPair(int[] source, int[] target)
    {
        var prepared = PrepareTarget(source, target);
        return new EncodedTextPair(prepared.Source, prepared.Target);
    }

    private (int[] Source, int[] Decoder, int[] Target) PrepareTarget(int[] source, int[] target)
    {
        var s = TokenSequences.Prefix(source, SourceLength, VocabularySize);
        var t = TokenSequences.Prefix(target, TargetLength, VocabularySize);
        if (t[^1] != CharacterTokenizer.EosId || t.Contains(CharacterTokenizer.BosId) || t.Take(t.Length - 1).Contains(CharacterTokenizer.EosId))
            throw new ArgumentException("Target must end with its only EOS and contain no BOS.", nameof(target));
        return (s, [CharacterTokenizer.BosId, .. t.Take(t.Length - 1)], t);
    }
}
