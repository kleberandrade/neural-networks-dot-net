namespace NeuralNetworks.Internal;

internal static class TransformerLayers
{
    public static Scalar[][] Embed(Scalar[] input, Scalar[] parameters, LinearBlock projection, int length, int features, int size, bool[]? padding = null)
    {
        var tokens = new Scalar[length][];
        for (var t = 0; t < length; t++)
        {
            if (padding is not null && padding[t]) { tokens[t] = Operations.Zeros(input[0], size); continue; }
            tokens[t] = projection.Apply(parameters, input.AsSpan(t * features, features).ToArray());
            AddPosition(tokens[t], t);
        }
        return tokens;
    }

    public static void AddPosition(Scalar[] token, int position)
    {
        for (var i = 0; i < token.Length; i++)
        {
            var angle = position / Math.Pow(10000, 2.0 * (i / 2) / token.Length);
            token[i] += i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
        }
    }

    public static Scalar[][] Residual(Scalar[][] input, Scalar[][] update, Scalar[] parameters, LayerNormalization norm) =>
        input.Select((token, t) => norm.Apply(parameters, token.Select((value, i) => value + update[t][i]).ToArray())).ToArray();

    public static bool[] Padding(bool[]? mask, int length, string name)
    {
        if (mask is null) return new bool[length];
        if (mask.Length != length || mask[0]) throw new ArgumentException("Provide a nonempty valid prefix, followed only by padding.", name);
        var seenPadding = false;
        foreach (var value in mask)
        {
            if (seenPadding && !value) throw new ArgumentException("Padding must be a trailing suffix.", name);
            seenPadding |= value;
        }
        return (bool[])mask.Clone();
    }

    public static double[][][] Snapshot(Scalar[][][] weights) =>
        weights.Select(head => head.Select(row => row.Select(x => x.Value).ToArray()).ToArray()).ToArray();
}

internal sealed record TransformerFeedForward(LinearBlock Input, LinearBlock Output, LayerNormalization Norm)
{
    public Scalar[][] Apply(Scalar[][] tokens, Scalar[] parameters)
    {
        var updates = tokens.Select(t => Output.Apply(parameters, Input.Apply(parameters, t).Select(x => x.Relu()).ToArray())).ToArray();
        return TransformerLayers.Residual(tokens, updates, parameters, Norm);
    }
}

internal readonly record struct TransformerEncoderLayer(MultiHeadAttention Attention, LayerNormalization Norm, TransformerFeedForward FeedForward)
{
    public AttentionResult Apply(Scalar[][] tokens, Scalar[] parameters, bool[]? padding = null)
    {
        var attention = Attention.Apply(tokens, tokens, parameters, padding);
        var normalized = TransformerLayers.Residual(tokens, attention.Output, parameters, Norm);
        return new AttentionResult(FeedForward.Apply(normalized, parameters), attention.Weights);
    }
}
