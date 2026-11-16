using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>Fixed-length, sequence-to-vector models. State resets for each evaluation.</summary>
public abstract class SequenceNetwork : GradientNetwork
{
    private readonly LinearBlock output;
    public int SequenceLength { get; }
    public int Features { get; }
    public int HiddenSize { get; }

    private protected SequenceNetwork(int sequenceLength, int inputSize, int hiddenSize, int outputSize, int seed)
        : base(checked(Guard.Positive(sequenceLength, nameof(sequenceLength)) * Guard.Positive(inputSize, nameof(inputSize))), outputSize, seed)
    {
        SequenceLength = sequenceLength;
        Features = inputSize;
        HiddenSize = Guard.Positive(hiddenSize, nameof(hiddenSize));
        output = AddLinear(hiddenSize, outputSize);
    }

    private protected sealed override Scalar[] Forward(Scalar[] input, Scalar[] parameters)
    {
        var hidden = Operations.Zeros(input[0], HiddenSize);
        var memory = Operations.Zeros(input[0], HiddenSize);
        for (var step = 0; step < SequenceLength; step++)
        {
            var features = input.AsSpan(step * Features, Features).ToArray();
            (hidden, memory) = Step(features, hidden, memory, parameters);
        }
        return Operations.Sigmoid(output.Apply(parameters, hidden));
    }

    private protected abstract (Scalar[] Hidden, Scalar[] Memory) Step(Scalar[] input, Scalar[] hidden, Scalar[] memory, Scalar[] parameters);
}
