namespace NeuralNetworks;

internal static class Guard
{
    public static int Positive(int value, string name) => value > 0
        ? value : throw new ArgumentOutOfRangeException(name, "Dimension must be positive.");

    public static double Finite(double value) => double.IsFinite(value)
        ? value : throw new ArithmeticException("Non-finite network arithmetic; scale inputs or parameters.");

    public static void Vector(double[] values, int length, string name)
    {
        ArgumentNullException.ThrowIfNull(values, name);
        if (values.Length != length || values.Any(x => !double.IsFinite(x)))
            throw new ArgumentException($"Expected {length} finite values.", name);
    }

    public static void Target(double[] target, int length)
    {
        Vector(target, length, nameof(target));
        if (target.Any(x => x < 0 || x > 1))
            throw new ArgumentException("Targets must be in [0, 1].", nameof(target));
    }

    public static void LearningRate(double rate)
    {
        if (!double.IsFinite(rate) || rate <= 0)
            throw new ArgumentOutOfRangeException(nameof(rate), "Learning rate must be finite and positive.");
    }

    public static double SquaredDistance(double[] a, double[] b)
    {
        var sum = 0.0;
        for (var i = 0; i < a.Length; i++)
        {
            var difference = Finite(a[i] - b[i]);
            sum = Finite(sum + Finite(difference * difference));
        }
        return sum;
    }
}
