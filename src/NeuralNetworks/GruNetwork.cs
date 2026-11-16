using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>A reset-before GRU; update gate interpolates between old state and candidate.</summary>
public sealed class GruNetwork : SequenceNetwork
{
    private readonly LinearBlock update;
    private readonly LinearBlock reset;
    private readonly LinearBlock candidate;

    public GruNetwork(int sequenceLength, int inputSize, int hiddenSize, int outputSize, int seed = 0)
        : base(sequenceLength, inputSize, hiddenSize, outputSize, seed)
    {
        var combined = checked(inputSize + hiddenSize);
        update = AddLinear(combined, hiddenSize);
        reset = AddLinear(combined, hiddenSize);
        candidate = AddLinear(combined, hiddenSize);
    }

    private protected override (Scalar[] Hidden, Scalar[] Memory) Step(Scalar[] input, Scalar[] hidden, Scalar[] memory, Scalar[] parameters)
    {
        Scalar[] combined = [.. input, .. hidden];
        var z = Operations.Sigmoid(update.Apply(parameters, combined));
        var r = Operations.Sigmoid(reset.Apply(parameters, combined));
        var resetHidden = new Scalar[HiddenSize];
        for (var unit = 0; unit < HiddenSize; unit++) resetHidden[unit] = r[unit] * hidden[unit];
        var proposed = Operations.Tanh(candidate.Apply(parameters, [.. input, .. resetHidden]));
        var next = new Scalar[HiddenSize];
        for (var unit = 0; unit < HiddenSize; unit++)
            next[unit] = z[unit] * hidden[unit] + (1 - z[unit]) * proposed[unit];
        return (next, memory);
    }
}
