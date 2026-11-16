namespace NeuralNetworks;

/// <summary>One-based successful-update schedule: optional linear warmup, then linear decay to a positive floor.</summary>
public sealed record LinearWarmupDecay(long WarmupSteps = 0, long DecaySteps = 1000, double MinimumFactor = 0.1)
{
    public double GetLearningRate(double baseLearningRate, long step)
    {
        Validate(); Guard.LearningRate(baseLearningRate);
        if (step <= 0) throw new ArgumentOutOfRangeException(nameof(step));
        double factor;
        if (WarmupSteps > 0 && step <= WarmupSteps) factor = step / (double)WarmupSteps;
        else
        {
            var peakStep = Math.Max(1, WarmupSteps);
            var progress = Math.Min((step - peakStep) / (double)DecaySteps, 1);
            factor = Math.Max(MinimumFactor, 1 - (1 - MinimumFactor) * progress);
        }
        var rate = Guard.Finite(baseLearningRate * factor);
        if (rate <= 0) throw new ArithmeticException("Scheduled learning rate underflowed to zero.");
        return rate;
    }

    internal void Validate()
    {
        if (WarmupSteps < 0 || DecaySteps <= 0 || !double.IsFinite(MinimumFactor) || MinimumFactor <= 0 || MinimumFactor > 1)
            throw new ArgumentOutOfRangeException(nameof(LinearWarmupDecay), "Use warmup >= 0, decay > 0 and minimum factor in (0, 1].");
    }
}
