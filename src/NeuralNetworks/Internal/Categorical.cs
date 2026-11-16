namespace NeuralNetworks.Internal;

internal static class Categorical
{
    public static double[] Probabilities(Scalar[] logits)
    {
        var shift = logits.Max(x => x.Value);
        var exponentials = logits.Select(x => Math.Exp(x.Value - shift)).ToArray();
        var sum = exponentials.Sum();
        return exponentials.Select(x => x / sum).ToArray();
    }

    public static Scalar CrossEntropy(Scalar[] logits, int target)
    {
        var shift = logits.Max(x => x.Value);
        var shifted = logits.Select(x => x + (-shift)).ToArray();
        // log-sum-exp avoids log(softmax[target]) underflow for very unlikely targets.
        return Operations.Sum(shifted.Select(x => x.Exp()).ToArray()).Log() - shifted[target];
    }
}
