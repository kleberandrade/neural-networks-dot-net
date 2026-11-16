using NeuralNetworks;

var tests = new (string Name, Action Run)[]
{
    ("Known forward pass and loss", Forward),
    ("Backprop matches central finite differences", Gradient),
    ("Gradient descent reduces loss", Descent),
    ("Seed and parameter snapshots are reproducible and isolated", Isolation),
    ("XOR learns all four cases", Xor),
    ("Invalid boundaries reject before mutation", Invalid),
    ("Extreme arithmetic is rejected atomically", Overflow),
    ("Perceptron learns AND and rejects invalid inputs", ClassicTests.PerceptronLearns),
    ("Hopfield recalls a noisy memory and decreases energy", ClassicTests.HopfieldRecalls),
    ("SOM updates winner and grid neighbors", ClassicTests.SomOrganizes),
    ("All differentiable architectures match numerical gradients", GradientModelTests.Gradients),
    ("New model boundaries and isolation", GradientModelTests.BoundariesAndIsolation),
    ("Known RBF and autoencoder arithmetic", GradientModelTests.KnownRbfAndAutoencoder),
    ("Known convolution and recurrent gate arithmetic", GradientModelTests.KnownSequenceAndConvolution),
    ("Known attention scores, normalization and cross-position influence", AttentionTests.KnownScaledAttention),
    ("All new differentiable architectures learn small tasks", GradientModelTests.LearnsTinyTasks),
    ("Independent multi-head scaled attention scores", TransformerTests.MultiHeadScores),
    ("Multi-head encoder and encoder-decoder numerical gradients", TransformerTests.Gradients),
    ("Decoder causality, padding and cross-attention", TransformerTests.CausalAndPaddingMasks),
    ("Encoder-decoder boundaries, isolation and atomic updates", TransformerTests.Boundaries),
    ("Encoder-decoder learns autoregressive sequence reversal", TransformerTests.LearnsReverse),
    ("Unicode character tokenization and special tokens", TextTransformerTests.Tokenizer),
    ("Deep numerical and token Transformer gradients", TextTransformerTests.StacksAndGradients),
    ("Categorical loss, stable softmax, EOS and generation cap", TextTransformerTests.CategoricalAndGeneration),
    ("Deep token causality, padding, validation and isolation", TextTransformerTests.CausalityAndBoundaries),
    ("Two-layer encoder-decoder learns text generation", TextTransformerTests.LearnsText),
    ("UTF-8 pair corpus and text continuation windows", CorpusCheckpointTests.CorpusFiles),
    ("Model/tokenizer checkpoint roundtrip and continued SGD", CorpusCheckpointTests.CheckpointRoundtrip),
    ("Corrupt, incompatible and oversized checkpoints reject", CorpusCheckpointTests.InvalidCheckpoints),
    ("Seeded temperature/top-k sampling", CorpusCheckpointTests.Sampling),
    ("File-backed epoch resume matches uninterrupted training", CorpusCheckpointTests.EpochResume),
    ("Adam known moments and bias-corrected updates", AdamTests.KnownUpdates),
    ("Minibatch mean gradients and final partial batch", AdamTests.MinibatchMean),
    ("Adam validation, ownership and atomic rejection", AdamTests.Boundaries),
    ("Adam checkpoint resume and legacy SGD compatibility", AdamTests.CheckpointResume),
    ("Global norm clipping and extreme finite gradients", TrainingControlTests.GlobalClipping),
    ("Warmup and linear decay known rates", TrainingControlTests.ScheduleValues),
    ("Clipped mean gradients, scheduled steps and atomic rejection", TrainingControlTests.BatchAndAtomicity),
    ("Scheduled clipped Adam resume and v2 compatibility", TrainingControlTests.CheckpointResume)
};
var failures = 0;
foreach (var test in tests)
{
    try { test.Run(); Console.WriteLine($"PASS {test.Name}"); }
    catch (Exception error) { failures++; Console.Error.WriteLine($"FAIL {test.Name}: {error}"); }
}
Console.WriteLine($"{tests.Length - failures}/{tests.Length} tests passed.");
return failures == 0 ? 0 : 1;

static void Near(double expected, double actual, double tolerance = 1e-9)
{
    if (!double.IsFinite(actual) || Math.Abs(expected - actual) > tolerance)
        throw new Exception($"Expected {expected}, got {actual} (tolerance {tolerance}).");
}
static void Throws<T>(Action action) where T : Exception
{
    try { action(); }
    catch (T) { return; }
    throw new Exception($"Expected {typeof(T).Name}.");
}
static void Same(double[] expected, double[] actual)
{
    if (!expected.SequenceEqual(actual)) throw new Exception("Arrays differ.");
}
static void Forward()
{
    var net = new DenseNetwork([2, 1], 1);
    net.SetParameters([0.5, -0.5, 0.25]);
    var output = 1 / (1 + Math.Exp(-0.75));
    Near(output, net.Predict([2, 1])[0]);
    Near(0.5 * Math.Pow(output - 1, 2), net.Loss([2, 1], [1]));
    var multilayer = new DenseNetwork([2, 2, 1], 0);
    multilayer.SetParameters([0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]);
    var hidden0 = 1 / (1 + Math.Exp(-0.3));
    var hidden1 = 1 / (1 + Math.Exp(1.9));
    Near(1 / (1 + Math.Exp(-(0.7 * hidden0 - 0.8 * hidden1 + 0.9))), multilayer.Predict([2, -1])[0]);
    net.SetParameters([0, 0, -1000]);
    Near(0, net.Predict([1, 1])[0]);
    net.SetParameters([0, 0, 1000]);
    Near(1, net.Predict([1, 1])[0]);
}
static void Gradient()
{
    foreach (var shape in new[] { new[] { 2, 2 }, new[] { 2, 3, 2, 2 } })
    {
        var net = new DenseNetwork(shape, 42);
        double[] input = [0.3, -0.7], target = [0.2, 0.8];
        var parameters = net.GetParameters();
        var analytic = net.ComputeGradient(input, target);
        for (var i = 0; i < parameters.Length; i++)
        {
            var perturbed = (double[])parameters.Clone();
            perturbed[i] += 1e-5;
            net.SetParameters(perturbed);
            var plus = net.Loss(input, target);
            perturbed[i] -= 2e-5;
            net.SetParameters(perturbed);
            Near((plus - net.Loss(input, target)) / 2e-5, analytic[i], 1e-8);
        }
        net.SetParameters(parameters);
    }
}
static void Descent()
{
    var net = new DenseNetwork([2, 3, 1], 5);
    var before = net.Loss([0.2, 0.8], [1]);
    Near(before, net.TrainSample([0.2, 0.8], [1], 0.5));
    if (net.Loss([0.2, 0.8], [1]) >= before) throw new Exception("Loss did not decrease.");
}
static void Isolation()
{
    int[] shape = [2, 3, 1];
    var a = new DenseNetwork(shape, 123);
    var b = new DenseNetwork(shape, 123);
    shape[0] = 100;
    Same(a.GetParameters(), b.GetParameters());
    var snapshot = a.GetParameters();
    snapshot[0] = 999;
    Same(a.Predict([0, 1]), b.Predict([0, 1]));
    a.SetParameters(snapshot);
    snapshot[0] = -999;
    Near(999, a.GetParameters()[0]);
    var prediction = b.Predict([0, 1]);
    prediction[0] = 999;
    if (b.Predict([0, 1])[0] == 999) throw new Exception("Prediction was aliased.");
}
static void Xor()
{
    double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    double[][] targets = [[0], [1], [1], [0]];
    var net = new DenseNetwork([2, 4, 1], 42);
    var initial = inputs.Select((x, i) => net.Loss(x, targets[i])).Average();
    for (var epoch = 0; epoch < 10000; epoch++)
        for (var i = 0; i < inputs.Length; i++) net.TrainSample(inputs[i], targets[i], 0.5);
    for (var i = 0; i < inputs.Length; i++) Near(targets[i][0], net.Predict(inputs[i])[0], 0.08);
    var final = inputs.Select((x, i) => net.Loss(x, targets[i])).Average();
    if (final >= initial / 20) throw new Exception("Insufficient loss reduction.");
}
static void Invalid()
{
    Throws<ArgumentNullException>(() => new DenseNetwork(null!, 0));
    Throws<ArgumentException>(() => new DenseNetwork([2], 0));
    Throws<ArgumentException>(() => new DenseNetwork([2, 0], 0));
    Throws<ArgumentException>(() => new DenseNetwork([-2, 1], 0));
    Throws<OverflowException>(() => new DenseNetwork([int.MaxValue, 1], 0));
    var net = new DenseNetwork([2, 1], 0);
    var before = net.GetParameters();
    Throws<ArgumentException>(() => net.Predict([1]));
    Throws<ArgumentNullException>(() => net.Predict(null!));
    Throws<ArgumentException>(() => net.Predict([double.NaN, 0]));
    Throws<ArgumentException>(() => net.Predict([double.PositiveInfinity, 0]));
    Throws<ArgumentException>(() => net.TrainSample([0, 1], [2], 0.5));
    Throws<ArgumentException>(() => net.Loss([0, 1], []));
    Throws<ArgumentException>(() => net.ComputeGradient([0, 1], [-0.1]));
    Throws<ArgumentException>(() => net.ComputeGradient([0, 1], [double.NaN]));
    Throws<ArgumentNullException>(() => net.Loss([0, 1], null!));
    foreach (var rate in new[] { 0, -1, double.NaN, double.PositiveInfinity })
        Throws<ArgumentOutOfRangeException>(() => net.TrainSample([0, 1], [1], rate));
    Throws<ArgumentException>(() => net.SetParameters([1]));
    Throws<ArgumentException>(() => net.SetParameters([0, double.NaN, 0]));
    Throws<ArgumentNullException>(() => net.SetParameters(null!));
    Same(before, net.GetParameters());
}
static void Overflow()
{
    var net = new DenseNetwork([2, 1], 0);
    net.SetParameters([double.MaxValue, double.MaxValue, 0]);
    var before = net.GetParameters();
    Throws<ArithmeticException>(() => net.TrainSample([2, 2], [0], 0.5));
    Same(before, net.GetParameters());
    net.SetParameters([0, 0, 0]);
    before = net.GetParameters();
    Throws<ArithmeticException>(() => net.TrainSample([0, double.MaxValue], [1], double.MaxValue));
    Same(before, net.GetParameters());
}
