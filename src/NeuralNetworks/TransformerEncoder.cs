using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>A stack of bidirectional multi-head encoder blocks with residuals, layer normalization and ReLU FFN.</summary>
public sealed class TransformerEncoder : GradientNetwork
{
    private readonly LinearBlock projection;
    private readonly LinearBlock output;
    private readonly TransformerEncoderLayer[] encoders;
    public int SequenceLength { get; }
    public int Features { get; }
    public int ModelSize { get; }
    public int HeadCount { get; }
    public int LayerCount { get; }

    // headCount is appended to preserve existing positional constructor calls and single-head snapshots.
    public TransformerEncoder(int sequenceLength, int inputSize, int modelSize, int outputSize, int feedForwardSize = 4, int seed = 0, int headCount = 1, int layerCount = 1)
        : base(checked(Guard.Positive(sequenceLength, nameof(sequenceLength)) * Guard.Positive(inputSize, nameof(inputSize))), outputSize, seed)
    {
        MultiHeadAttention.Validate(modelSize, headCount);
        Guard.Positive(feedForwardSize, nameof(feedForwardSize));
        SequenceLength = sequenceLength;
        Features = inputSize;
        ModelSize = modelSize;
        HeadCount = headCount;
        LayerCount = Guard.Positive(layerCount, nameof(layerCount));
        encoders = new TransformerEncoderLayer[layerCount];
        projection = AddLinear(inputSize, modelSize);
        var attention = new MultiHeadAttention(AddLinear(modelSize, modelSize), AddLinear(modelSize, modelSize),
            AddLinear(modelSize, modelSize), AddLinear(modelSize, modelSize), modelSize, headCount);
        var feedIn = AddLinear(modelSize, feedForwardSize);
        var feedOut = AddLinear(feedForwardSize, modelSize);
        output = AddLinear(modelSize, outputSize);
        encoders[0] = new TransformerEncoderLayer(attention, AddNormalization(modelSize),
            new TransformerFeedForward(feedIn, feedOut, AddNormalization(modelSize)));
        var builder = new TransformerBuilder(AddLinear, AddNormalization, modelSize, headCount, feedForwardSize);
        for (var layer = 1; layer < LayerCount; layer++) encoders[layer] = builder.Encoder();
    }

    /// <summary>Returns one head, indexed by query then key; the default is head zero.</summary>
    public double[][] GetAttentionWeights(double[] input, int headIndex = 0, int layerIndex = 0)
    {
        if (headIndex < 0 || headIndex >= HeadCount) throw new ArgumentOutOfRangeException(nameof(headIndex));
        return GetMultiHeadAttentionWeights(input, layerIndex)[headIndex];
    }

    public double[][][] GetMultiHeadAttentionWeights(double[] input, int layerIndex = 0)
    {
        if (layerIndex < 0 || layerIndex >= LayerCount) throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return GetLayerAttentionWeights(input)[layerIndex];
    }

    public double[][][][] GetLayerAttentionWeights(double[] input)
    {
        var graph = CreateGraph(input);
        var tokens = TransformerLayers.Embed(graph.Input, graph.Parameters, projection, SequenceLength, Features, ModelSize);
        return TransformerStacks.Snapshot(TransformerStacks.Encode(tokens, graph.Parameters, encoders).Weights);
    }

    private protected override Scalar[] Forward(Scalar[] input, Scalar[] parameters)
    {
        var tokens = TransformerLayers.Embed(input, parameters, projection, SequenceLength, Features, ModelSize);
        var encoded = TransformerStacks.Encode(tokens, parameters, encoders).Output;
        var pooled = Operations.Zeros(input[0], ModelSize);
        foreach (var token in encoded)
            for (var i = 0; i < ModelSize; i++) pooled[i] += token[i] / SequenceLength;
        return Operations.Sigmoid(output.Apply(parameters, pooled));
    }
}
