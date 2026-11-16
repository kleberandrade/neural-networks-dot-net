namespace NeuralNetworks;

public sealed record AdamOptions(double Beta1 = 0.9, double Beta2 = 0.999, double Epsilon = 1e-8);
public sealed record AdamState(AdamOptions Options, long Step, double[] FirstMoment, double[] SecondMoment);

/// <summary>Model-bound Adam with bias correction and transactional parameter/moment updates.</summary>
public sealed class AdamOptimizer
{
    private readonly ParameterizedModel model;
    private readonly AdamOptions options;
    private double[] firstMoment;
    private double[] secondMoment;
    private long step;

    public AdamOptimizer(ParameterizedModel model, AdamOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        this.model = model; this.options = options ?? new AdamOptions(); ValidateOptions(this.options);
        firstMoment = new double[model.ParameterCount]; secondMoment = new double[model.ParameterCount];
    }

    public AdamState GetState() => new(options, step, (double[])firstMoment.Clone(), (double[])secondMoment.Clone());

    public static AdamOptimizer Restore(ParameterizedModel model, AdamState state)
    {
        ArgumentNullException.ThrowIfNull(model); ValidateState(state, model.ParameterCount);
        var optimizer = new AdamOptimizer(model, state.Options)
        {
            step = state.Step,
            firstMoment = (double[])state.FirstMoment.Clone(),
            secondMoment = (double[])state.SecondMoment.Clone()
        };
        return optimizer;
    }

    internal static void ValidateState(AdamState state, int parameterCount)
    {
        ArgumentNullException.ThrowIfNull(state); ValidateOptions(state.Options);
        if (state.Step < 0) throw new ArgumentOutOfRangeException(nameof(state));
        Guard.Vector(state.FirstMoment, parameterCount, nameof(state.FirstMoment));
        Guard.Vector(state.SecondMoment, parameterCount, nameof(state.SecondMoment));
        if (state.SecondMoment.Any(x => x < 0) || (state.Step == 0 && (state.FirstMoment.Any(x => x != 0) || state.SecondMoment.Any(x => x != 0))))
            throw new ArgumentException("Invalid Adam moment state.", nameof(state));
    }

    private static void ValidateOptions(AdamOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!double.IsFinite(options.Beta1) || options.Beta1 < 0 || options.Beta1 >= 1
            || !double.IsFinite(options.Beta2) || options.Beta2 < 0 || options.Beta2 >= 1
            || !double.IsFinite(options.Epsilon) || options.Epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Adam requires beta values in [0, 1) and finite positive epsilon.");
    }

    public void Step(double[] gradient, double learningRate)
    {
        Guard.Vector(gradient, model.ParameterCount, nameof(gradient)); Guard.LearningRate(learningRate);
        var nextStep = checked(step + 1); var updated = model.GetParameters();
        var nextFirst = new double[updated.Length]; var nextSecond = new double[updated.Length];
        var firstCorrection = 1 - Math.Pow(options.Beta1, nextStep);
        var secondCorrection = 1 - Math.Pow(options.Beta2, nextStep);
        for (var i = 0; i < updated.Length; i++)
        {
            nextFirst[i] = Guard.Finite(options.Beta1 * firstMoment[i] + (1 - options.Beta1) * gradient[i]);
            nextSecond[i] = Guard.Finite(options.Beta2 * secondMoment[i] + (1 - options.Beta2) * Guard.Finite(gradient[i] * gradient[i]));
            var correctedFirst = Guard.Finite(nextFirst[i] / firstCorrection);
            var correctedSecond = Guard.Finite(nextSecond[i] / secondCorrection);
            var denominator = Guard.Finite(Math.Sqrt(correctedSecond) + options.Epsilon);
            var direction = Guard.Finite(correctedFirst / denominator);
            updated[i] = Guard.Finite(updated[i] - Guard.Finite(learningRate * direction));
        }
        model.SetParameters(updated);
        firstMoment = nextFirst; secondMoment = nextSecond; step = nextStep;
    }

    internal void EnsureModel(ParameterizedModel expected)
    {
        if (!ReferenceEquals(model, expected)) throw new ArgumentException("Adam belongs to a different model instance.");
    }

}
