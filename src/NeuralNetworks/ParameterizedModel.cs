using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>Shared parameter ownership and initialization for the educational numerical and token models.</summary>
public abstract class ParameterizedModel
{
    private double[] parameters = [];
    private readonly Random random;
    public int ParameterCount => parameters.Length;
    private protected ParameterizedModel(int seed) => random = new Random(seed);
    public double[] GetParameters() => (double[])parameters.Clone();

    public void SetParameters(double[] values)
    {
        Guard.Vector(values, ParameterCount, nameof(values));
        parameters = (double[])values.Clone();
    }

    private protected LinearBlock AddLinear(int inputs, int outputs)
    {
        Guard.Positive(inputs, nameof(inputs)); Guard.Positive(outputs, nameof(outputs));
        var offset = AddParameters(new double[checked(outputs * checked(inputs + 1))]);
        var limit = Math.Sqrt(6.0 / ((double)inputs + outputs));
        for (var neuron = 0; neuron < outputs; neuron++)
            for (var i = 0; i < inputs; i++)
                parameters[offset + neuron * (inputs + 1) + i] = (2 * random.NextDouble() - 1) * limit;
        return new LinearBlock(inputs, outputs, offset);
    }

    private protected EmbeddingBlock AddEmbedding(int vocabularySize, int size)
    {
        var offset = AddParameters(new double[checked(vocabularySize * size)]);
        var limit = Math.Sqrt(6.0 / ((double)vocabularySize + size));
        for (var i = offset; i < parameters.Length; i++) parameters[i] = (2 * random.NextDouble() - 1) * limit;
        return new EmbeddingBlock(offset, size);
    }

    private protected LayerNormalization AddNormalization(int size) =>
        new(size, AddParameters([.. Enumerable.Repeat(1.0, size), .. new double[size]]));

    private protected int AddParameters(double[] values)
    {
        var offset = parameters.Length;
        Array.Resize(ref parameters, checked(offset + values.Length)); values.CopyTo(parameters, offset);
        return offset;
    }

    private protected (Tape Tape, Scalar[] Parameters) CreateParameterGraph()
    {
        var tape = new Tape();
        return (tape, parameters.Select(tape.Leaf).ToArray());
    }

    private protected static double[] Derivatives(Scalar loss, Scalar[] parameterNodes)
    {
        var derivatives = loss.Tape.Backward(loss);
        return parameterNodes.Select(p => derivatives[p.Id]).ToArray();
    }

    private protected void ApplyUpdate(double[] gradient, double learningRate)
    {
        var updated = new double[ParameterCount];
        for (var i = 0; i < updated.Length; i++)
            updated[i] = Guard.Finite(parameters[i] - Guard.Finite(learningRate * gradient[i]));
        parameters = updated;
    }
}
