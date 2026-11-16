using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>One bidirectional multi-head encoder block with residuals, layer normalization and ReLU FFN.</summary>
public sealed class TransformerEncoder : GradientNetwork
{
    private readonly LinearBlock projection;
    private readonly LinearBlock output;
    private readonly TransformerEncoderLayer encoder;
    public int SequenceLength { get; }
    public int Features { get; }
    public int ModelSize { get; }
    public int HeadCount { get; }

    // headCount is appended to preserve existing positional constructor calls and single-head snapshots.
    public TransformerEncoder(int sequenceLength, int inputSize, int modelSize, int outputSize, int feedForwardSize = 4, int seed = 0, int headCount = 1)
        : base(checked(Guard.Positive(sequenceLength, nameof(sequenceLength)) * Guard.Positive(inputSize, nameof(inputSize))), outputSize, seed)
    {
        MultiHeadAttention.Validate(modelSize, headCount);
        Guard.Positive(feedForwardSize, nameof(feedForwardSize));
        SequenceLength = sequenceLength;
        Features = inputSize;
        ModelSize = modelSize;
        HeadCount = headCount;
        projection = AddLinear(inputSize, modelSize);
        var attention = new MultiHeadAttention(AddLinear(modelSize, modelSize), AddLinear(modelSize, modelSize),
            AddLinear(modelSize, modelSize), AddLinear(modelSize, modelSize), modelSize, headCount);
        var feedIn = AddLinear(modelSize, feedForwardSize);
        var feedOut = AddLinear(feedForwardSize, modelSize);
        output = AddLinear(modelSize, outputSize);
        encoder = new TransformerEncoderLayer(attention, AddNormalization(modelSize),
            new TransformerFeedForward(feedIn, feedOut, AddNormalization(modelSize)));
    }

    /// <summary>Returns one head, indexed by query then key; the default is head zero.</summary>
    public double[][] GetAttentionWeights(double[] input, int headIndex = 0)
    {
        if (headIndex < 0 || headIndex >= HeadCount) throw new ArgumentOutOfRangeException(nameof(headIndex));
        return GetMultiHeadAttentionWeights(input)[headIndex];
    }

    public double[][][] GetMultiHeadAttentionWeights(double[] input)
    {
        var graph = CreateGraph(input);
        var tokens = TransformerLayers.Embed(graph.Input, graph.Parameters, projection, SequenceLength, Features, ModelSize);
        return TransformerLayers.Snapshot(encoder.Attention.Apply(tokens, tokens, graph.Parameters).Weights);
    }

    private protected override Scalar[] Forward(Scalar[] input, Scalar[] parameters)
    {
        var tokens = TransformerLayers.Embed(input, parameters, projection, SequenceLength, Features, ModelSize);
        var encoded = encoder.Apply(tokens, parameters).Output;
        var pooled = Operations.Zeros(input[0], ModelSize);
        foreach (var token in encoded)
            for (var i = 0; i < ModelSize; i++) pooled[i] += token[i] / SequenceLength;
        return Operations.Sigmoid(output.Apply(parameters, pooled));
    }
}
