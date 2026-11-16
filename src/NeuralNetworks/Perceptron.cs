namespace NeuralNetworks;

/// <summary>A binary linear classifier trained with the perceptron mistake rule.</summary>
public sealed class Perceptron
{
    private double[] parameters;
    public int InputSize { get; }

    public Perceptron(int inputSize)
    {
        InputSize = Guard.Positive(inputSize, nameof(inputSize));
        parameters = new double[checked(inputSize + 1)];
    }

    public double[] GetParameters() => (double[])parameters.Clone();

    public void SetParameters(double[] values)
    {
        Guard.Vector(values, parameters.Length, nameof(values));
        parameters = (double[])values.Clone();
    }

    public int Predict(double[] input)
    {
        Guard.Vector(input, InputSize, nameof(input));
        var score = parameters[^1];
        for (var i = 0; i < InputSize; i++)
            score = Guard.Finite(score + Guard.Finite(parameters[i] * input[i]));
        return score >= 0 ? 1 : 0;
    }

    /// <summary>Applies the mistake update and returns target minus the pre-update prediction.</summary>
    public int TrainSample(double[] input, int target, double learningRate)
    {
        if (target is not (0 or 1)) throw new ArgumentException("Target must be 0 or 1.", nameof(target));
        Guard.LearningRate(learningRate);
        var error = target - Predict(input);
        var updated = GetParameters();
        for (var i = 0; i < InputSize; i++)
            updated[i] = Guard.Finite(updated[i] + Guard.Finite(learningRate * error * input[i]));
        updated[^1] = Guard.Finite(updated[^1] + learningRate * error);
        parameters = updated;
        return error;
    }
}
