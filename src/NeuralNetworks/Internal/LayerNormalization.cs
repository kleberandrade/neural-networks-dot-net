namespace NeuralNetworks.Internal;

internal readonly record struct LayerNormalization(int Size, int Offset)
{
    public Scalar[] Apply(Scalar[] parameters, Scalar[] input)
    {
        var mean = Operations.Sum(input) / Size;
        var variance = input[0].Constant(0);
        foreach (var value in input)
        {
            var difference = value - mean;
            variance += difference * difference / Size;
        }
        var inverseDeviation = (variance + 1e-5).Sqrt().Reciprocal();
        var result = new Scalar[Size];
        for (var i = 0; i < Size; i++)
            result[i] = (input[i] - mean) * inverseDeviation * parameters[Offset + i] + parameters[Offset + Size + i];
        return result;
    }
}
