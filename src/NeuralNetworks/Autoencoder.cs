using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>A sigmoid encoder and decoder trained to reconstruct inputs in [0, 1].</summary>
public sealed class Autoencoder : GradientNetwork
{
    private readonly LinearBlock encoder;
    private readonly LinearBlock decoder;
    public int LatentSize { get; }

    public Autoencoder(int inputSize, int latentSize, int seed = 0) : base(inputSize, inputSize, seed)
    {
        LatentSize = Guard.Positive(latentSize, nameof(latentSize));
        if (latentSize >= inputSize) throw new ArgumentException("Use a bottleneck smaller than the input.", nameof(latentSize));
        encoder = AddLinear(inputSize, latentSize);
        decoder = AddLinear(latentSize, inputSize);
    }

    public double[] Encode(double[] input)
    {
        var graph = CreateGraph(input);
        return Operations.Sigmoid(encoder.Apply(graph.Parameters, graph.Input)).Select(x => x.Value).ToArray();
    }

    public double TrainSample(double[] input, double learningRate) => base.TrainSample(input, input, learningRate);

    private protected override Scalar[] Forward(Scalar[] input, Scalar[] parameters) =>
        Operations.Sigmoid(decoder.Apply(parameters, Operations.Sigmoid(encoder.Apply(parameters, input))));
}
