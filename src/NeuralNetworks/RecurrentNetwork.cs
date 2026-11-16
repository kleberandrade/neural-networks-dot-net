using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>An Elman tanh RNN with full backpropagation through the fixed-length sequence.</summary>
public sealed class RecurrentNetwork : SequenceNetwork
{
    private readonly LinearBlock cell;

    public RecurrentNetwork(int sequenceLength, int inputSize, int hiddenSize, int outputSize, int seed = 0)
        : base(sequenceLength, inputSize, hiddenSize, outputSize, seed) =>
        cell = AddLinear(checked(inputSize + hiddenSize), hiddenSize);

    private protected override (Scalar[] Hidden, Scalar[] Memory) Step(Scalar[] input, Scalar[] hidden, Scalar[] memory, Scalar[] parameters) =>
        (Operations.Tanh(cell.Apply(parameters, [.. input, .. hidden])), memory);
}
