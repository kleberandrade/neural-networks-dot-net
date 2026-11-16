using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>An LSTM with input, forget, candidate and output gates; no peepholes.</summary>
public sealed class LstmNetwork : SequenceNetwork
{
    private readonly LinearBlock forget;
    private readonly LinearBlock inputGate;
    private readonly LinearBlock candidate;
    private readonly LinearBlock outputGate;

    public LstmNetwork(int sequenceLength, int inputSize, int hiddenSize, int outputSize, int seed = 0)
        : base(sequenceLength, inputSize, hiddenSize, outputSize, seed)
    {
        var combined = checked(inputSize + hiddenSize);
        forget = AddLinear(combined, hiddenSize);
        inputGate = AddLinear(combined, hiddenSize);
        candidate = AddLinear(combined, hiddenSize);
        outputGate = AddLinear(combined, hiddenSize);
    }

    private protected override (Scalar[] Hidden, Scalar[] Memory) Step(Scalar[] input, Scalar[] hidden, Scalar[] memory, Scalar[] parameters)
    {
        Scalar[] combined = [.. input, .. hidden];
        var f = Operations.Sigmoid(forget.Apply(parameters, combined));
        var i = Operations.Sigmoid(inputGate.Apply(parameters, combined));
        var g = Operations.Tanh(candidate.Apply(parameters, combined));
        var o = Operations.Sigmoid(outputGate.Apply(parameters, combined));
        var nextMemory = new Scalar[HiddenSize];
        var nextHidden = new Scalar[HiddenSize];
        for (var unit = 0; unit < HiddenSize; unit++)
        {
            nextMemory[unit] = f[unit] * memory[unit] + i[unit] * g[unit];
            nextHidden[unit] = o[unit] * nextMemory[unit].Tanh();
        }
        return (nextHidden, nextMemory);
    }
}
