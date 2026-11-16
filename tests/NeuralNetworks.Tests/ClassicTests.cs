using NeuralNetworks;

internal static class ClassicTests
{
    public static void PerceptronLearns()
    {
        var net = new Perceptron(2);
        double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        int[] targets = [0, 0, 0, 1];
        for (var epoch = 0; epoch < 20; epoch++)
            for (var i = 0; i < inputs.Length; i++) net.TrainSample(inputs[i], targets[i], 0.1);
        for (var i = 0; i < inputs.Length; i++) Check.Near(targets[i], net.Predict(inputs[i]));
        net.SetParameters([0.5, -0.5, 0.25]);
        Check.Near(1, net.Predict([2, 1]));
        Check.Near(0, net.Predict([0, 1]));
        var before = net.GetParameters();
        Check.Throws<ArgumentException>(() => net.TrainSample([0, 1], 2, 0.1));
        Check.Throws<ArgumentException>(() => net.Predict([double.NaN, 0]));
        Check.Throws<ArgumentOutOfRangeException>(() => net.TrainSample([0, 1], 1, 0));
        Check.Throws<ArgumentException>(() => net.SetParameters([1]));
        Check.Throws<ArgumentOutOfRangeException>(() => new Perceptron(0));
        Check.Same(before, net.GetParameters());
        net.SetParameters([0, 0, 0]);
        Check.Throws<ArithmeticException>(() => net.TrainSample([0, double.MaxValue], 0, double.MaxValue));
        Check.Same([0, 0, 0], net.GetParameters());
    }

    public static void HopfieldRecalls()
    {
        double[] pattern = [-1, -1, -1, -1, 1, 1, 1, 1];
        var net = new HopfieldNetwork(pattern.Length);
        Check.Throws<InvalidOperationException>(() => net.Recall(pattern));
        net.Fit([pattern]);
        var noisy = (double[])pattern.Clone();
        noisy[1] *= -1;
        var recalled = net.Recall(noisy);
        Check.True(recalled.Converged, "Recall did not reach a fixed point.");
        Check.Same(pattern, recalled.Pattern);
        Check.True(net.Energy(recalled.Pattern) <= net.Energy(noisy), "Energy increased.");
        Check.Near(-3.5, net.Energy(pattern));
        Check.Same(pattern, net.Recall(pattern).Pattern);
        var before = net.Recall(noisy).Pattern;
        Check.Throws<ArgumentException>(() => net.Fit([pattern, [1, 0]]));
        Check.Throws<ArgumentException>(() => net.Fit([]));
        Check.Throws<ArgumentException>(() => net.Recall([0, 0, 0, 0, 0, 0, 0, 0]));
        Check.Throws<ArgumentOutOfRangeException>(() => net.Recall(pattern, 0));
        Check.Same(before, net.Recall(noisy).Pattern);
        pattern[0] = 1;
        Check.Near(-3.5, net.Energy(recalled.Pattern));
    }

    public static void SomOrganizes()
    {
        var net = new SelfOrganizingMap(2, 2, 2, seed: 42);
        var duplicate = new SelfOrganizingMap(2, 2, 2, seed: 42);
        Check.Same(net.GetWeights(), duplicate.GetWeights());
        double[] input = [0.8, 0.2];
        var before = net.GetWeights();
        var winner = net.FindBestMatchingUnit(input);
        net.TrainSample(input, learningRate: 0.5, radius: 1);
        var after = net.GetWeights();
        for (var unit = 0; unit < 4; unit++)
        {
            var dx = unit % 2 - winner % 2;
            var dy = unit / 2 - winner / 2;
            var influence = Math.Exp(-(dx * dx + dy * dy) / 2.0);
            for (var feature = 0; feature < 2; feature++)
                Check.Near(before[unit * 2 + feature] + 0.5 * influence * (input[feature] - before[unit * 2 + feature]), after[unit * 2 + feature]);
        }
        var single = new SelfOrganizingMap(1, 1, 2, 9);
        single.TrainSample(input, 1, 0);
        Check.Same(input, single.GetWeights());
        var saved = net.GetWeights();
        Check.Throws<ArgumentException>(() => net.TrainSample([double.NaN, 0], 0.5, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => net.TrainSample(input, 2, 1));
        Check.Throws<ArgumentOutOfRangeException>(() => net.TrainSample(input, 0.5, -1));
        Check.Throws<ArithmeticException>(() => net.TrainSample([double.MaxValue, 0], 0.5, 1));
        Check.Same(saved, net.GetWeights());
        saved[0] = 999;
        Check.True(net.GetWeights()[0] != 999, "Weights were aliased.");
    }
}
