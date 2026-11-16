namespace NeuralNetworks.Internal;

internal readonly record struct EmbeddingBlock(int Offset, int Size)
{
    public Scalar[][] Apply(int[] tokens, Scalar[] parameters)
    {
        var result = new Scalar[tokens.Length][];
        for (var t = 0; t < tokens.Length; t++)
        {
            result[t] = new Scalar[Size];
            for (var i = 0; i < Size; i++) result[t][i] = parameters[Offset + tokens[t] * Size + i];
            TransformerLayers.AddPosition(result[t], t);
        }
        return result;
    }
}
