namespace NeuralNetworks.Internal;

internal readonly record struct LinearBlock(int Inputs, int Outputs, int Offset)
{
    public Scalar[] Apply(Scalar[] parameters, Scalar[] input)
    {
        var result = new Scalar[Outputs];
        for (var neuron = 0; neuron < Outputs; neuron++)
        {
            var start = Offset + neuron * (Inputs + 1);
            var sum = parameters[start + Inputs];
            for (var i = 0; i < Inputs; i++) sum += parameters[start + i] * input[i];
            result[neuron] = sum;
        }
        return result;
    }
}

internal static class Operations
{
    public static Scalar[] Zeros(Scalar anchor, int count) =>
        Enumerable.Range(0, count).Select(_ => anchor.Constant(0)).ToArray();

    public static Scalar[] Sigmoid(Scalar[] input) => input.Select(x => x.Sigmoid()).ToArray();
    public static Scalar[] Tanh(Scalar[] input) => input.Select(x => x.Tanh()).ToArray();

    public static Scalar Sum(Scalar[] values)
    {
        var sum = values[0].Constant(0);
        foreach (var value in values) sum += value;
        return sum;
    }

    public static Scalar Dot(Scalar[] a, Scalar[] b)
    {
        var sum = a[0].Constant(0);
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }
}
