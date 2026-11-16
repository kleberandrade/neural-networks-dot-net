namespace NeuralNetworks;

/// <summary>A small dense feedforward network with sigmoid activations and squared error.</summary>
/// <remarks>Instances are mutable and are not safe for concurrent training or parameter updates.</remarks>
public sealed class DenseNetwork
{
    private readonly int[] sizes;
    private readonly int[] offsets;
    private double[] parameters;

    /// <summary>Creates a network with Xavier uniform weights and zero biases.</summary>
    public DenseNetwork(int[] layerSizes, int seed = 0)
    {
        ArgumentNullException.ThrowIfNull(layerSizes);
        if (layerSizes.Length < 2 || layerSizes.Any(size => size <= 0))
            throw new ArgumentException("Provide at least two layers, each with a positive size.", nameof(layerSizes));
        sizes = (int[])layerSizes.Clone();
        offsets = new int[sizes.Length];
        for (var layer = 1; layer < sizes.Length; layer++)
            offsets[layer] = checked(offsets[layer - 1] + checked(sizes[layer] * checked(sizes[layer - 1] + 1)));
        parameters = new double[offsets[^1]];
        Initialize(new Random(seed));
    }

    public int InputSize => sizes[0];
    public int OutputSize => sizes[^1];
    public int ParameterCount => parameters.Length;

    /// <summary>Returns a detached parameter snapshot in layer/neuron/weight/bias order.</summary>
    public double[] GetParameters() => (double[])parameters.Clone();

    /// <summary>Copies a complete finite parameter vector into the network.</summary>
    public void SetParameters(double[] values)
    {
        ValidateVector(values, ParameterCount, nameof(values));
        parameters = (double[])values.Clone();
    }

    /// <summary>Computes sigmoid outputs without changing the model.</summary>
    public double[] Predict(double[] input) => Forward(input)[^1];

    /// <summary>Returns one half of the sum of squared output errors.</summary>
    public double Loss(double[] input, double[] target)
    {
        ValidateTarget(target);
        return SquaredError(Predict(input), target);
    }

    /// <summary>Returns the loss gradient in the same order as GetParameters, without updating parameters.</summary>
    public double[] ComputeGradient(double[] input, double[] target)
    {
        ValidateTarget(target);
        return Backpropagate(Forward(input), target);
    }

    /// <summary>Applies one stochastic gradient step and returns the loss before the update.</summary>
    /// <remarks>Invalid arguments or non-finite arithmetic leave the model unchanged.</remarks>
    public double TrainSample(double[] input, double[] target, double learningRate)
    {
        if (!double.IsFinite(learningRate) || learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be finite and positive.");
        ValidateTarget(target);
        var activations = Forward(input);
        var gradient = Backpropagate(activations, target);
        var updated = new double[ParameterCount];
        for (var i = 0; i < updated.Length; i++)
            updated[i] = Finite(parameters[i] - learningRate * gradient[i]);
        parameters = updated;
        return SquaredError(activations[^1], target);
    }

    private void Initialize(Random random)
    {
        for (var layer = 1; layer < sizes.Length; layer++)
        {
            var limit = Math.Sqrt(6.0 / ((double)sizes[layer - 1] + sizes[layer]));
            for (var neuron = 0; neuron < sizes[layer]; neuron++)
                for (var input = 0; input < sizes[layer - 1]; input++)
                    parameters[Index(layer, neuron) + input] = (2 * random.NextDouble() - 1) * limit;
        }
    }

    private double[][] Forward(double[] input)
    {
        ValidateVector(input, InputSize, nameof(input));
        var activations = new double[sizes.Length][];
        activations[0] = (double[])input.Clone();
        for (var layer = 1; layer < sizes.Length; layer++)
            activations[layer] = ActivateLayer(layer, activations[layer - 1]);
        return activations;
    }

    private double[] ActivateLayer(int layer, double[] previous)
    {
        var result = new double[sizes[layer]];
        for (var neuron = 0; neuron < result.Length; neuron++)
        {
            var start = Index(layer, neuron);
            var sum = parameters[start + previous.Length];
            for (var input = 0; input < previous.Length; input++)
                sum = Finite(sum + Finite(parameters[start + input] * previous[input]));
            result[neuron] = Sigmoid(sum);
        }
        return result;
    }

    private double[] Backpropagate(double[][] activations, double[] target)
    {
        var gradient = new double[ParameterCount];
        var delta = new double[OutputSize];
        for (var neuron = 0; neuron < delta.Length; neuron++)
        {
            var output = activations[^1][neuron];
            delta[neuron] = (output - target[neuron]) * output * (1 - output);
        }
        for (var layer = sizes.Length - 1; layer > 0; layer--)
        {
            WriteGradient(gradient, layer, activations[layer - 1], delta);
            if (layer > 1) delta = HiddenDelta(layer, activations[layer - 1], delta);
        }
        return gradient;
    }

    private void WriteGradient(double[] gradient, int layer, double[] previous, double[] delta)
    {
        for (var neuron = 0; neuron < delta.Length; neuron++)
        {
            var start = Index(layer, neuron);
            for (var input = 0; input < previous.Length; input++)
                gradient[start + input] = Finite(delta[neuron] * previous[input]);
            gradient[start + previous.Length] = delta[neuron];
        }
    }

    private double[] HiddenDelta(int layer, double[] previous, double[] delta)
    {
        var result = new double[previous.Length];
        for (var input = 0; input < result.Length; input++)
        {
            var sum = 0.0;
            for (var neuron = 0; neuron < delta.Length; neuron++)
                sum = Finite(sum + Finite(parameters[Index(layer, neuron) + input] * delta[neuron]));
            result[input] = Finite(sum * previous[input] * (1 - previous[input]));
        }
        return result;
    }

    private int Index(int layer, int neuron) => offsets[layer - 1] + neuron * (sizes[layer - 1] + 1);

    private static double Sigmoid(double value)
    {
        // Both branches avoid exp(large positive), which would overflow.
        var exponential = Math.Exp(-Math.Abs(value));
        return value >= 0 ? 1 / (1 + exponential) : exponential / (1 + exponential);
    }

    private static double SquaredError(double[] output, double[] target)
    {
        var sum = 0.0;
        for (var i = 0; i < output.Length; i++)
            sum += 0.5 * Math.Pow(output[i] - target[i], 2);
        return sum;
    }

    private void ValidateTarget(double[] target)
    {
        ValidateVector(target, OutputSize, nameof(target));
        if (target.Any(value => value < 0 || value > 1))
            throw new ArgumentException("Sigmoid targets must be in [0, 1].", nameof(target));
    }

    private static void ValidateVector(double[] values, int length, string name)
    {
        ArgumentNullException.ThrowIfNull(values, name);
        if (values.Length != length || values.Any(value => !double.IsFinite(value)))
            throw new ArgumentException($"Expected {length} finite values.", name);
    }

    private static double Finite(double value) => double.IsFinite(value)
        ? value
        : throw new ArithmeticException("Network arithmetic overflowed. Scale inputs, weights or the learning rate.");
}
