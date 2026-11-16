namespace NeuralNetworks;

/// <summary>Global L2-norm clipping that preserves direction without squaring large raw coordinates.</summary>
public static class GradientClipping
{
    public static double[] ClipGlobalNorm(double[] gradient, double maxNorm)
    {
        ArgumentNullException.ThrowIfNull(gradient); Guard.Vector(gradient, gradient.Length, nameof(gradient));
        ValidateThreshold(maxNorm);
        if (gradient.Length == 0) throw new ArgumentException("Provide a nonempty gradient.", nameof(gradient));
        var maximum = gradient.Max(x => Math.Abs(x));
        if (maximum == 0) return (double[])gradient.Clone();
        var scaledNorm = Math.Sqrt(gradient.Sum(x => (x / maximum) * (x / maximum)));
        if (maximum <= maxNorm / scaledNorm) return (double[])gradient.Clone();
        // Divide before multiplying: both the raw norm and maxNorm / rawNorm may overflow/underflow.
        var radius = maxNorm / scaledNorm;
        return gradient.Select(x => (x / maximum) * radius).ToArray();
    }

    internal static void ValidateThreshold(double maxNorm)
    {
        if (!double.IsFinite(maxNorm) || maxNorm <= 0) throw new ArgumentOutOfRangeException(nameof(maxNorm));
    }
}
