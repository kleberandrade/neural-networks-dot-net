using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>Fixed Gaussian radial basis centers and width with a trainable sigmoid output head.</summary>
public sealed class RadialBasisNetwork : GradientNetwork
{
    private readonly double[][] centers;
    private readonly double width;
    private readonly LinearBlock output;

    public RadialBasisNetwork(double[][] centers, double width, int outputSize, int seed = 0)
        : base(CenterSize(centers), outputSize, seed)
    {
        if (!double.IsFinite(width) || width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        foreach (var center in centers) Guard.Vector(center, InputSize, nameof(centers));
        this.centers = centers.Select(x => (double[])x.Clone()).ToArray();
        this.width = width;
        output = AddLinear(centers.Length, outputSize);
    }

    private protected override Scalar[] Forward(Scalar[] input, Scalar[] parameters)
    {
        var basis = new Scalar[centers.Length];
        for (var center = 0; center < centers.Length; center++)
        {
            var distance = input[0].Constant(0);
            for (var feature = 0; feature < InputSize; feature++)
            {
                var normalized = (input[feature] + (-centers[center][feature])) / width;
                distance += normalized * normalized;
            }
            basis[center] = (distance * -0.5).Exp();
        }
        return Operations.Sigmoid(output.Apply(parameters, basis));
    }

    private static int CenterSize(double[][] centers)
    {
        ArgumentNullException.ThrowIfNull(centers);
        if (centers.Length == 0 || centers[0] is null || centers[0].Length == 0)
            throw new ArgumentException("Provide at least one nonempty finite center.", nameof(centers));
        return centers[0].Length;
    }
}
