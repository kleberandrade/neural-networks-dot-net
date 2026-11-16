using NeuralNetworks;

internal static class GradientModelTests
{
    public static (string Name, Func<GradientNetwork> Create, double[] Input)[] Cases =>
    [
        ("RBF", () => new RadialBasisNetwork([[0, 0], [1, 1]], 0.7, 2, 7), [0.2, 0.8]),
        ("autoencoder", () => new Autoencoder(2, 1, 7), [0.2, 0.8]),
        ("CNN", () => new ConvolutionalNetwork(3, 3, 2, 2, 2, 7), [0.1, 0.3, 0.7, 0.2, 0.4, 0.8, 0.3, 0.5, 0.9]),
        ("RNN", () => new RecurrentNetwork(3, 1, 2, 2, 7), [0.2, -0.3, 0.7]),
        ("LSTM", () => new LstmNetwork(3, 1, 2, 2, 7), [0.2, -0.3, 0.7]),
        ("GRU", () => new GruNetwork(3, 1, 2, 2, 7), [0.2, -0.3, 0.7]),
        ("Transformer", () => new TransformerEncoder(2, 1, 3, 2, 4, 7), [0.2, 0.7])
    ];

    public static void Gradients()
    {
        foreach (var item in Cases)
        {
            var model = item.Create();
            var target = Enumerable.Repeat(0.3, model.OutputSize).ToArray();
            var parameters = model.GetParameters();
            var gradient = model.ComputeGradient(item.Input, target);
            Check.Same(parameters, model.GetParameters());
            Check.True(gradient.Length == parameters.Length, item.Name + " gradient shape");
            for (var i = 0; i < parameters.Length; i++)
            {
                var changed = (double[])parameters.Clone();
                changed[i] += 1e-5;
                model.SetParameters(changed);
                var plus = model.Loss(item.Input, target);
                changed[i] -= 2e-5;
                model.SetParameters(changed);
                var numeric = (plus - model.Loss(item.Input, target)) / 2e-5;
                try { Check.Near(numeric, gradient[i], 2e-6); }
                catch (Exception error) { throw new Exception($"{item.Name} parameter {i}: {error.Message}"); }
            }
            model.SetParameters(parameters);
            var before = model.Loss(item.Input, target);
            Check.Near(before, model.TrainSample(item.Input, target, 0.01));
            Check.True(model.Loss(item.Input, target) < before, item.Name + " descent failed");
        }
    }

    public static void BoundariesAndIsolation()
    {
        foreach (var item in Cases)
        {
            var a = item.Create();
            var b = item.Create();
            var before = a.GetParameters();
            Check.Same(before, b.GetParameters());
            var target = Enumerable.Repeat(0.3, a.OutputSize).ToArray();
            Check.Same(a.Predict(item.Input), b.Predict(item.Input));
            Check.Throws<ArgumentNullException>(() => a.Predict(null!));
            Check.Throws<ArgumentException>(() => a.Predict([]));
            var invalid = (double[])item.Input.Clone();
            invalid[0] = double.PositiveInfinity;
            Check.Throws<ArgumentException>(() => a.Predict(invalid));
            Check.Throws<ArgumentException>(() => a.TrainSample(item.Input, [], 0.1));
            Check.Throws<ArgumentException>(() => a.TrainSample(item.Input, Enumerable.Repeat(2.0, a.OutputSize).ToArray(), 0.1));
            Check.Throws<ArgumentOutOfRangeException>(() => a.TrainSample(item.Input, target, double.NaN));
            Check.Throws<ArgumentException>(() => a.SetParameters([]));
            var invalidParameters = (double[])before.Clone();
            invalidParameters[^1] = double.NaN;
            Check.Throws<ArgumentException>(() => a.SetParameters(invalidParameters));
            Check.Same(before, a.GetParameters());
            var snapshot = a.GetParameters();
            a.SetParameters(snapshot);
            snapshot[0] = 999;
            Check.Same(before, a.GetParameters());
            var output = a.Predict(item.Input);
            output[0] = 999;
            Check.True(a.Predict(item.Input)[0] != 999, item.Name + " prediction was aliased");
            a.TrainSample(item.Input, target, 0.1);
            b.TrainSample(item.Input, target, 0.1);
            Check.Same(a.GetParameters(), b.GetParameters());
            var huge = Enumerable.Repeat(double.MaxValue, a.ParameterCount).ToArray();
            a.SetParameters(huge);
            Check.Throws<ArithmeticException>(() => a.TrainSample(item.Input, target, 0.1));
            Check.Same(huge, a.GetParameters());
        }
        Check.Throws<ArgumentOutOfRangeException>(() => new RecurrentNetwork(0, 1, 2, 1));
        Check.Throws<ArgumentException>(() => new ConvolutionalNetwork(2, 2, 3, 1, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => new LstmNetwork(2, 1, 0, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => new GruNetwork(2, 1, 2, 0));
        Check.Throws<ArgumentException>(() => new Autoencoder(2, 2));
        Check.Throws<ArgumentException>(() => new RadialBasisNetwork([], 1, 1));
        Check.Throws<ArgumentException>(() => new RadialBasisNetwork([[0, 1], [0]], 1, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => new RadialBasisNetwork([[0]], 0, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => new TransformerEncoder(2, 1, 1, 1));
    }

    public static void KnownRbfAndAutoencoder()
    {
        double[][] centers = [[0], [1]];
        var rbf = new RadialBasisNetwork(centers, 1, 1);
        rbf.SetParameters([1, -1, 0]);
        var expected = 1 / (1 + Math.Exp(-(1 - Math.Exp(-0.5))));
        Check.Near(expected, rbf.Predict([0])[0]);
        centers[0][0] = 999;
        Check.Near(expected, rbf.Predict([0])[0]);
        var auto = new Autoencoder(2, 1);
        auto.SetParameters([0.5, -0.5, 0, 1, 0, -1, 0]);
        Check.Near(0.5, auto.Encode([1, 1])[0]);
        Check.Near(1 / (1 + Math.Exp(-0.5)), auto.Predict([1, 1])[0]);
    }

    public static void KnownSequenceAndConvolution()
    {
        static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        var cnn = new ConvolutionalNetwork(3, 3, 2, 1, 1);
        cnn.SetParameters([1, 0, 0, 0, 0, 1, 0]);
        Check.Near(Sigmoid(3), cnn.Predict([1, 2, 3, 4, 5, 6, 7, 8, 9])[0]);
        var relu = new ConvolutionalNetwork(2, 2, 2, 1, 1);
        relu.SetParameters([1, 0, 0, -1, 0, 2, -1]);
        Check.Near(Sigmoid(-1), relu.Predict([1, 2, 3, 4])[0]);
        Check.Near(Sigmoid(5), relu.Predict([4, 2, 3, 1])[0]);
        var rnn = new RecurrentNetwork(2, 1, 1, 1);
        rnn.SetParameters([1, 0, 1, 0.5, 0]);
        Check.Near(Sigmoid(Math.Tanh(0.5 * Math.Tanh(1))), rnn.Predict([1, 0])[0]);
        var lstm = new LstmNetwork(2, 1, 1, 1);
        lstm.SetParameters([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
        Check.Near(Sigmoid(0.5 * Math.Tanh(0.25 * Math.Tanh(1))), lstm.Predict([1, 0])[0]);
        var gru = new GruNetwork(2, 1, 1, 1);
        gru.SetParameters([1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]);
        var h1 = 0.5 * Math.Tanh(1);
        Check.Near(Sigmoid(0.5 * h1 + 0.5 * Math.Tanh(0.5 * h1)), gru.Predict([1, 0])[0]);
    }

    public static void LearnsTinyTasks()
    {
        foreach (var (name, model, inputs, targets) in LearningCases())
        {
            var initial = inputs.Select((x, i) => model.Loss(x, targets[i])).Average();
            for (var epoch = 0; epoch < 1200; epoch++)
                for (var i = 0; i < inputs.Length; i++) model.TrainSample(inputs[i], targets[i], 0.5);
            var final = inputs.Select((x, i) => model.Loss(x, targets[i])).Average();
            Console.WriteLine($"  {name}: mean loss {initial:F6} -> {final:F6}");
            Check.True(final < initial / 5, name + " insufficient learning");
            for (var i = 0; i < inputs.Length; i++)
                for (var j = 0; j < targets[i].Length; j++)
                    Check.Near(targets[i][j], model.Predict(inputs[i])[j], 0.2);
        }
    }

    private static IEnumerable<(string, GradientNetwork, double[][], double[][])> LearningCases()
    {
        yield return ("RBF XOR", new RadialBasisNetwork([[0, 0], [0, 1], [1, 0], [1, 1]], 0.4, 1, 42), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]);
        yield return ("autoencoder reconstruction", new Autoencoder(2, 1, 42), [[0, 1], [1, 0]], [[0, 1], [1, 0]]);
        yield return ("CNN intensity", new ConvolutionalNetwork(2, 2, 2, 2, 1, 42), [[-1, -1, -1, -1], [1, 1, 1, 1]], [[0], [1]]);
        yield return ("RNN memory", new RecurrentNetwork(3, 1, 3, 1, 42), [[0, 0, 0], [1, 0, 0]], [[0], [1]]);
        yield return ("LSTM memory", new LstmNetwork(3, 1, 3, 1, 42), [[0, 0, 0], [1, 0, 0]], [[0], [1]]);
        yield return ("GRU memory", new GruNetwork(3, 1, 3, 1, 42), [[0, 0, 0], [1, 0, 0]], [[0], [1]]);
        yield return ("Transformer intensity", new TransformerEncoder(2, 1, 4, 1, 4, 42), [[0, 0], [1, 1]], [[0], [1]]);
    }
}
