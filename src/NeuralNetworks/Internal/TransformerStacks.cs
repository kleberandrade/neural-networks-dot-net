namespace NeuralNetworks.Internal;

internal readonly record struct DecoderResult(Scalar[][] Output, Scalar[][][] Self, Scalar[][][] Cross);
internal sealed record TransformerDecoderLayer(MultiHeadAttention Self, LayerNormalization SelfNorm,
    MultiHeadAttention Cross, LayerNormalization CrossNorm, TransformerFeedForward FeedForward)
{
    public DecoderResult Apply(Scalar[][] tokens, Scalar[][] memory, Scalar[] parameters, bool[]? sourcePadding, bool[]? targetPadding)
    {
        var self = Self.Apply(tokens, tokens, parameters, targetPadding, causal: true);
        var first = TransformerLayers.Residual(tokens, self.Output, parameters, SelfNorm);
        var cross = Cross.Apply(first, memory, parameters, sourcePadding);
        var second = TransformerLayers.Residual(first, cross.Output, parameters, CrossNorm);
        return new DecoderResult(FeedForward.Apply(second, parameters), self.Weights, cross.Weights);
    }
}

/// <summary>Allocates fresh, independent parameter blocks for every requested layer.</summary>
internal sealed class TransformerBuilder(Func<int, int, LinearBlock> linear, Func<int, LayerNormalization> norm,
    int modelSize, int headCount, int feedForwardSize)
{
    public MultiHeadAttention Attention() => new(linear(modelSize, modelSize), linear(modelSize, modelSize),
        linear(modelSize, modelSize), linear(modelSize, modelSize), modelSize, headCount);
    public TransformerFeedForward FeedForward() => new(linear(modelSize, feedForwardSize), linear(feedForwardSize, modelSize), norm(modelSize));
    public TransformerEncoderLayer Encoder()
    {
        var attention = Attention(); var normalization = norm(modelSize);
        return new TransformerEncoderLayer(attention, normalization, FeedForward());
    }
    public TransformerDecoderLayer Decoder()
    {
        var self = Attention(); var selfNorm = norm(modelSize);
        var cross = Attention(); var crossNorm = norm(modelSize);
        return new TransformerDecoderLayer(self, selfNorm, cross, crossNorm, FeedForward());
    }
}

internal readonly record struct StackResult(Scalar[][] Output, Scalar[][][][] Encoder, Scalar[][][][] Decoder, Scalar[][][][] Cross);
internal static class TransformerStacks
{
    public static (Scalar[][] Output, Scalar[][][][] Weights) Encode(Scalar[][] tokens, Scalar[] parameters,
        TransformerEncoderLayer[] layers, bool[]? padding = null)
    {
        var weights = new Scalar[layers.Length][][][];
        for (var i = 0; i < layers.Length; i++)
        {
            var result = layers[i].Apply(tokens, parameters, padding);
            tokens = result.Output; weights[i] = result.Weights;
        }
        return (tokens, weights);
    }

    public static StackResult Apply(Scalar[][] source, Scalar[][] target, Scalar[] parameters,
        TransformerEncoderLayer[] encoders, TransformerDecoderLayer[] decoders, bool[]? sourcePadding = null, bool[]? targetPadding = null)
    {
        var memory = Encode(source, parameters, encoders, sourcePadding);
        var self = new Scalar[decoders.Length][][][]; var cross = new Scalar[decoders.Length][][][];
        for (var i = 0; i < decoders.Length; i++)
        {
            var result = decoders[i].Apply(target, memory.Output, parameters, sourcePadding, targetPadding);
            target = result.Output; self[i] = result.Self; cross[i] = result.Cross;
        }
        return new StackResult(target, memory.Weights, self, cross);
    }

    public static double[][][][] Snapshot(Scalar[][][][] weights) => weights.Select(TransformerLayers.Snapshot).ToArray();
}
