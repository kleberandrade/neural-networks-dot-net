namespace NeuralNetworks;

/// <summary>A rectangular Kohonen map with Euclidean winners and Gaussian grid neighborhoods.</summary>
public sealed class SelfOrganizingMap
{
    private double[] weights;
    public int Width { get; }
    public int Height { get; }
    public int InputSize { get; }

    public SelfOrganizingMap(int width, int height, int inputSize, int seed = 0)
    {
        Width = Guard.Positive(width, nameof(width));
        Height = Guard.Positive(height, nameof(height));
        InputSize = Guard.Positive(inputSize, nameof(inputSize));
        weights = new double[checked(checked(width * height) * inputSize)];
        var random = new Random(seed);
        for (var i = 0; i < weights.Length; i++) weights[i] = random.NextDouble();
    }

    public double[] GetWeights() => (double[])weights.Clone();

    /// <summary>Returns row-major unit index; ties select the first unit.</summary>
    public int FindBestMatchingUnit(double[] input)
    {
        Guard.Vector(input, InputSize, nameof(input));
        var best = double.PositiveInfinity;
        var winner = 0;
        for (var unit = 0; unit < Width * Height; unit++)
        {
            var distance = 0.0;
            for (var feature = 0; feature < InputSize; feature++)
            {
                var difference = Guard.Finite(input[feature] - weights[unit * InputSize + feature]);
                distance = Guard.Finite(distance + Guard.Finite(difference * difference));
            }
            if (distance < best) { best = distance; winner = unit; }
        }
        return winner;
    }

    /// <summary>Returns the pre-update winner. Radius zero updates only the winner.</summary>
    public int TrainSample(double[] input, double learningRate, double radius)
    {
        Guard.LearningRate(learningRate);
        if (learningRate > 1) throw new ArgumentOutOfRangeException(nameof(learningRate), "Use a rate in (0, 1].");
        if (!double.IsFinite(radius) || radius < 0) throw new ArgumentOutOfRangeException(nameof(radius));
        var winner = FindBestMatchingUnit(input);
        var updated = GetWeights();
        for (var unit = 0; unit < Width * Height; unit++)
        {
            var dx = (double)(unit % Width - winner % Width);
            var dy = (double)(unit / Width - winner / Width);
            var gridDistance = dx * dx + dy * dy;
            var influence = radius == 0 ? (unit == winner ? 1 : 0) : Math.Exp(-0.5 * (gridDistance / radius) / radius);
            for (var feature = 0; feature < InputSize; feature++)
            {
                var index = unit * InputSize + feature;
                // A convex combination avoids an unnecessary overflow in (input - weight).
                var amount = learningRate * influence;
                updated[index] = Guard.Finite((1 - amount) * weights[index] + amount * input[feature]);
            }
        }
        weights = updated;
        return winner;
    }
}
