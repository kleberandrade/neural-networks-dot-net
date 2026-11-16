using NeuralNetworks.Internal;

namespace NeuralNetworks;

/// <summary>Common squared-error SGD API for the small tape-backed educational models.</summary>
/// <remarks>Mutable, allocation-heavy, and not safe for concurrent mutation. Not an extension API.</remarks>
public abstract class GradientNetwork : IGradientNetwork
{
    private double[] parameters = [];
    private readonly Random random;
    public int InputSize { get; }
    public int OutputSize { get; }
    public int ParameterCount => parameters.Length;

    private protected GradientNetwork(int inputSize, int outputSize, int seed)
    {
        InputSize = Guard.Positive(inputSize, nameof(inputSize));
        OutputSize = Guard.Positive(outputSize, nameof(outputSize));
        random = new Random(seed);
    }

    public double[] GetParameters() => (double[])parameters.Clone();

    public void SetParameters(double[] values)
    {
        Guard.Vector(values, ParameterCount, nameof(values));
        parameters = (double[])values.Clone();
    }

    public double[] Predict(double[] input) => PredictUsing(input, Forward);
    public double Loss(double[] input, double[] target) => LossUsing(input, target, Forward);
    public double[] ComputeGradient(double[] input, double[] target) => GradientUsing(input, target, Forward);
    public double TrainSample(double[] input, double[] target, double learningRate) =>
        TrainUsing(input, target, learningRate, Forward);

    private protected double[] PredictUsing(double[] input, Func<Scalar[], Scalar[], Scalar[]> forward) =>
        Evaluate(input, forward).Output.Select(x => x.Value).ToArray();

    private protected double LossUsing(double[] input, double[] target, Func<Scalar[], Scalar[], Scalar[]> forward, bool[]? mask = null)
    {
        Guard.Target(target, OutputSize);
        return Objective(Evaluate(input, forward).Output, target, mask).Value;
    }

    private protected double[] GradientUsing(double[] input, double[] target, Func<Scalar[], Scalar[], Scalar[]> forward, bool[]? mask = null)
    {
        Guard.Target(target, OutputSize);
        return Gradient(Evaluate(input, forward), target, mask).Values;
    }

    private protected double TrainUsing(double[] input, double[] target, double learningRate, Func<Scalar[], Scalar[], Scalar[]> forward, bool[]? mask = null)
    {
        Guard.LearningRate(learningRate);
        Guard.Target(target, OutputSize);
        var gradient = Gradient(Evaluate(input, forward), target, mask);
        var updated = new double[ParameterCount];
        for (var i = 0; i < updated.Length; i++)
            updated[i] = Guard.Finite(parameters[i] - Guard.Finite(learningRate * gradient.Values[i]));
        parameters = updated;
        return gradient.Loss;
    }

    private protected LinearBlock AddLinear(int inputs, int outputs)
    {
        Guard.Positive(inputs, nameof(inputs));
        Guard.Positive(outputs, nameof(outputs));
        var offset = parameters.Length;
        Array.Resize(ref parameters, checked(offset + checked(outputs * checked(inputs + 1))));
        var limit = Math.Sqrt(6.0 / ((double)inputs + outputs));
        for (var neuron = 0; neuron < outputs; neuron++)
            for (var i = 0; i < inputs; i++)
                parameters[offset + neuron * (inputs + 1) + i] = (2 * random.NextDouble() - 1) * limit;
        return new LinearBlock(inputs, outputs, offset);
    }

    private protected LayerNormalization AddNormalization(int size) =>
        new(size, AddParameters([.. Enumerable.Repeat(1.0, size), .. new double[size]]));

    private protected int AddParameters(double[] values)
    {
        var offset = parameters.Length;
        Array.Resize(ref parameters, checked(offset + values.Length));
        values.CopyTo(parameters, offset);
        return offset;
    }

    private protected (Scalar[] Input, Scalar[] Parameters) CreateGraph(double[] input)
    {
        Guard.Vector(input, InputSize, nameof(input));
        var tape = new Tape();
        return (input.Select(tape.Leaf).ToArray(), parameters.Select(tape.Leaf).ToArray());
    }

    private (Scalar[] Output, Scalar[] Parameters) Evaluate(double[] input, Func<Scalar[], Scalar[], Scalar[]> forward)
    {
        var graph = CreateGraph(input);
        return (forward(graph.Input, graph.Parameters), graph.Parameters);
    }

    private static Scalar Objective(Scalar[] output, double[] target, bool[]? mask)
    {
        var loss = output[0].Constant(0);
        for (var i = 0; i < output.Length; i++)
        {
            if (mask is not null && mask[i]) continue;
            var error = output[i] + (-target[i]);
            loss += error * error * 0.5;
        }
        return loss;
    }

    private static (double Loss, double[] Values) Gradient((Scalar[] Output, Scalar[] Parameters) graph, double[] target, bool[]? mask)
    {
        var loss = Objective(graph.Output, target, mask);
        var gradients = loss.Tape.Backward(loss);
        return (loss.Value, graph.Parameters.Select(p => gradients[p.Id]).ToArray());
    }

    private protected abstract Scalar[] Forward(Scalar[] input, Scalar[] parameters);
}
