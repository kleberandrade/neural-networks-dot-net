namespace NeuralNetworks.Internal;

internal readonly record struct AttentionResult(Scalar[][] Output, Scalar[][][] Weights);

/// <summary>Full-width Q/K/V projections split into contiguous heads, then concatenated and projected.</summary>
internal sealed record MultiHeadAttention(LinearBlock Query, LinearBlock Key, LinearBlock Value,
    LinearBlock Output, int ModelSize, int HeadCount)
{
    public static void Validate(int modelSize, int headCount)
    {
        if (modelSize < 2) throw new ArgumentOutOfRangeException(nameof(modelSize), "Use at least two model dimensions.");
        Guard.Positive(headCount, nameof(headCount));
        if (modelSize % headCount != 0)
            throw new ArgumentException("Model size must be divisible by head count.", nameof(headCount));
    }

    public AttentionResult Apply(Scalar[][] queries, Scalar[][] memory, Scalar[] parameters, bool[]? keyPadding = null, bool causal = false)
    {
        var q = queries.Select(t => Query.Apply(parameters, t)).ToArray();
        var k = memory.Select(t => Key.Apply(parameters, t)).ToArray();
        var v = memory.Select(t => Value.Apply(parameters, t)).ToArray();
        var contexts = queries.Select(t => Operations.Zeros(t[0], ModelSize)).ToArray();
        var weights = new Scalar[HeadCount][][];
        var headSize = ModelSize / HeadCount;
        for (var head = 0; head < HeadCount; head++)
        {
            weights[head] = new Scalar[queries.Length][];
            for (var query = 0; query < queries.Length; query++)
            {
                var offset = head * headSize;
                var row = Row(q[query], k, offset, headSize, query, keyPadding, causal);
                weights[head][query] = row;
                for (var key = 0; key < memory.Length; key++)
                    for (var feature = 0; feature < headSize; feature++)
                        contexts[query][offset + feature] += row[key] * v[key][offset + feature];
            }
        }
        return new AttentionResult(contexts.Select(t => Output.Apply(parameters, t)).ToArray(), weights);
    }

    private static Scalar[] Row(Scalar[] query, Scalar[][] keys, int offset, int size, int position, bool[]? padding, bool causal)
    {
        var allowed = Enumerable.Range(0, keys.Length)
            .Where(i => (padding is null || !padding[i]) && (!causal || i <= position)).ToArray();
        if (allowed.Length == 0) throw new ArgumentException("Every attention query requires at least one visible key.");
        var scores = allowed.Select(i => Score(query, keys[i], offset, size)).ToArray();
        var shift = scores.Max(s => s.Value);
        var exponentials = scores.Select(s => (s + (-shift)).Exp()).ToArray();
        var inverseSum = Operations.Sum(exponentials).Reciprocal();
        var result = Operations.Zeros(query[0], keys.Length);
        for (var i = 0; i < allowed.Length; i++) result[allowed[i]] = exponentials[i] * inverseSum;
        return result;
    }

    private static Scalar Score(Scalar[] query, Scalar[] key, int offset, int size)
    {
        var sum = query[0].Constant(0);
        for (var i = offset; i < offset + size; i++) sum += query[i] * key[i];
        return sum / Math.Sqrt(size);
    }
}
