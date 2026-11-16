namespace NeuralNetworks;

public sealed record HopfieldRecall(double[] Pattern, bool Converged, int Sweeps);

/// <summary>A bipolar associative memory using Hebbian weights and asynchronous recall.</summary>
public sealed class HopfieldNetwork
{
    private double[]? weights;
    public int Size { get; }

    public HopfieldNetwork(int size) => Size = Guard.Positive(size, nameof(size));

    /// <summary>Replaces the memory bank; validates all patterns before changing weights.</summary>
    public void Fit(double[][] patterns)
    {
        ArgumentNullException.ThrowIfNull(patterns);
        if (patterns.Length == 0) throw new ArgumentException("Provide at least one pattern.", nameof(patterns));
        foreach (var pattern in patterns) ValidatePattern(pattern);
        var updated = new double[checked(Size * Size)];
        foreach (var pattern in patterns)
            for (var i = 0; i < Size; i++)
                for (var j = 0; j < i; j++)
                {
                    updated[i * Size + j] += pattern[i] * pattern[j] / Size;
                    updated[j * Size + i] = updated[i * Size + j];
                }
        weights = updated;
    }

    public HopfieldRecall Recall(double[] pattern, int maxSweeps = 100)
    {
        ValidatePattern(pattern);
        Guard.Positive(maxSweeps, nameof(maxSweeps));
        var memory = RequireWeights();
        var state = (double[])pattern.Clone();
        for (var sweep = 1; sweep <= maxSweeps; sweep++)
        {
            var changed = false;
            for (var i = 0; i < Size; i++)
            {
                var field = 0.0;
                for (var j = 0; j < Size; j++) field += memory[i * Size + j] * state[j];
                var next = field == 0 ? state[i] : Math.Sign(field);
                changed |= next != state[i];
                state[i] = next;
            }
            if (!changed) return new HopfieldRecall(state, true, sweep);
        }
        return new HopfieldRecall(state, false, maxSweeps);
    }

    public double Energy(double[] pattern)
    {
        ValidatePattern(pattern);
        var memory = RequireWeights();
        var energy = 0.0;
        for (var i = 0; i < Size; i++)
            for (var j = 0; j < i; j++) energy -= memory[i * Size + j] * pattern[i] * pattern[j];
        return energy;
    }

    private double[] RequireWeights() => weights ?? throw new InvalidOperationException("Fit a memory bank first.");

    private void ValidatePattern(double[] pattern)
    {
        Guard.Vector(pattern, Size, nameof(pattern));
        if (pattern.Any(value => value is not (-1 or 1)))
            throw new ArgumentException("Patterns must be bipolar (-1 or +1).", nameof(pattern));
    }
}
