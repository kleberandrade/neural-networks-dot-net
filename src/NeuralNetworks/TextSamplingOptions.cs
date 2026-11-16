using NeuralNetworks.Internal;

namespace NeuralNetworks;

public sealed record TextSamplingOptions(double Temperature = 1, int TopK = 0, int Seed = 0, int? MaxNewTokens = null);

internal static class TokenSampling
{
    public static void Validate(TextSamplingOptions options, int vocabularySize)
    {
        if (!double.IsFinite(options.Temperature) || options.Temperature <= 0) throw new ArgumentOutOfRangeException(nameof(options.Temperature));
        if (options.TopK < 0 || options.TopK > vocabularySize - 2) throw new ArgumentOutOfRangeException(nameof(options.TopK));
    }

    public static int Greedy(Scalar[] logits)
    {
        var next = CharacterTokenizer.EosId;
        for (var id = CharacterTokenizer.EosId + 1; id < logits.Length; id++)
            if (logits[id].Value > logits[next].Value) next = id;
        return next;
    }

    public static int Choose(Scalar[] logits, TextSamplingOptions options, Random random)
    {
        var candidates = Enumerable.Range(CharacterTokenizer.EosId, logits.Length - 2)
            .OrderByDescending(id => logits[id].Value).ThenBy(id => id)
            .Take(options.TopK == 0 ? logits.Length - 2 : options.TopK).ToArray();
        var maximum = logits[candidates[0]].Value;
        var weights = candidates.Select(id => Math.Exp((logits[id].Value - maximum) / options.Temperature)).ToArray();
        var draw = random.NextDouble() * weights.Sum(); var cumulative = 0.0;
        for (var i = 0; i < weights.Length; i++)
        {
            cumulative += weights[i];
            if (draw < cumulative) return candidates[i];
        }
        return candidates[^1];
    }
}
