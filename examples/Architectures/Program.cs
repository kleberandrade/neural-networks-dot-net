using System.Globalization;
using NeuralNetworks;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
var demos = new (string Name, Action Run)[]
{
    ("perceptron", PerceptronDemo),
    ("mlp", () => Train("MLP XOR", new DenseNetwork([2, 4, 1], 42), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], 10000)),
    ("rbf", () => Train("RBF XOR", new RadialBasisNetwork([[0, 0], [0, 1], [1, 0], [1, 1]], 0.4, 1, 42), [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])),
    ("hopfield", HopfieldDemo),
    ("som", SomDemo),
    ("cnn", () => Train("CNN centered intensity", new ConvolutionalNetwork(2, 2, 2, 2, 1, 42), [[-1, -1, -1, -1], [1, 1, 1, 1]], [[0], [1]])),
    ("rnn", () => MemoryDemo("RNN", new RecurrentNetwork(3, 1, 3, 1, 42))),
    ("lstm", () => MemoryDemo("LSTM", new LstmNetwork(3, 1, 3, 1, 42))),
    ("gru", () => MemoryDemo("GRU", new GruNetwork(3, 1, 3, 1, 42))),
    ("autoencoder", AutoencoderDemo),
    ("transformer", () => Train("Transformer intensity", new TransformerEncoder(2, 1, 4, 1, 4, 42), [[0, 0], [1, 1]], [[0], [1]]))
};
var choice = args.Length == 0 ? "all" : args[0];
if (args.Length > 1 || (choice != "all" && demos.All(x => x.Name != choice)))
{
    Console.Error.WriteLine("Usage: dotnet run --project examples/Architectures -- [all|" + string.Join('|', demos.Select(x => x.Name)) + "]");
    return 2;
}
var failures = 0;
foreach (var demo in demos.Where(x => choice == "all" || choice == x.Name))
{
    Console.WriteLine($"\n=== {demo.Name} ===");
    try { demo.Run(); }
    catch (Exception error) { failures++; Console.Error.WriteLine($"FAIL: {error.Message}"); }
}
return failures == 0 ? 0 : 1;

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

static string Format(double[] values) => "[" + string.Join(", ", values.Select(x => x.ToString("F4", CultureInfo.InvariantCulture))) + "]";

static void Train(string name, IGradientNetwork model, double[][] inputs, double[][] targets, int epochs = 1200)
{
    var initial = inputs.Select((x, i) => model.Loss(x, targets[i])).Average();
    for (var epoch = 0; epoch < epochs; epoch++)
        for (var i = 0; i < inputs.Length; i++) model.TrainSample(inputs[i], targets[i], 0.5);
    var final = inputs.Select((x, i) => model.Loss(x, targets[i])).Average();
    Console.WriteLine($"{name}: mean loss {initial:F6} -> {final:F6} ({epochs} epochs)");
    Require(final < initial / 5, "Mean loss did not fall to less than one fifth of its initial value.");
    for (var i = 0; i < inputs.Length; i++)
    {
        var output = model.Predict(inputs[i]);
        Console.WriteLine($"{Format(inputs[i])} -> {Format(output)}; target {Format(targets[i])}");
        Require(output.Zip(targets[i]).All(pair => Math.Abs(pair.First - pair.Second) < 0.2), "Prediction missed the example's tolerance.");
    }
}

static void MemoryDemo(string name, IGradientNetwork model) =>
    Train(name + " remembers the first feature after two zero steps", model, [[0, 0, 0], [1, 0, 0]], [[0], [1]]);

static void PerceptronDemo()
{
    var model = new Perceptron(2);
    double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    int[] targets = [0, 0, 0, 1];
    for (var epoch = 0; epoch < 20; epoch++)
        for (var i = 0; i < inputs.Length; i++) model.TrainSample(inputs[i], targets[i], 0.1);
    for (var i = 0; i < inputs.Length; i++)
    {
        var output = model.Predict(inputs[i]);
        Console.WriteLine($"{Format(inputs[i])} -> {output}; expected AND={targets[i]}");
        Require(output == targets[i], "Perceptron did not learn AND.");
    }
}

static void HopfieldDemo()
{
    double[] memory = [-1, -1, -1, -1, 1, 1, 1, 1];
    var model = new HopfieldNetwork(memory.Length);
    model.Fit([memory]);
    var noisy = (double[])memory.Clone();
    noisy[1] *= -1;
    var recalled = model.Recall(noisy);
    Console.WriteLine($"Noisy: {Format(noisy)}");
    Console.WriteLine($"Recall: {Format(recalled.Pattern)}; converged={recalled.Converged}; sweeps={recalled.Sweeps}");
    Console.WriteLine($"Energy: {model.Energy(noisy):F4} -> {model.Energy(recalled.Pattern):F4}");
    Require(recalled.Converged && recalled.Pattern.SequenceEqual(memory), "Memory was not recovered.");
}

static void SomDemo()
{
    var model = new SelfOrganizingMap(2, 2, 2, 42);
    double[][] samples = [[0.1, 0.1], [0.9, 0.9]];
    double Error() => samples.Select(x =>
    {
        var winner = model.FindBestMatchingUnit(x);
        var weights = model.GetWeights();
        return x.Select((value, i) => Math.Pow(value - weights[winner * model.InputSize + i], 2)).Sum();
    }).Average();
    var initial = Error();
    for (var epoch = 0; epoch < 200; epoch++)
    {
        var progress = epoch / 199.0;
        foreach (var sample in samples) model.TrainSample(sample, 0.5 * (1 - progress) + 0.05 * progress, 1 - progress);
    }
    var final = Error();
    Console.WriteLine($"Mean squared quantization distance: {initial:F6} -> {final:F6}");
    foreach (var sample in samples)
    {
        var winner = model.FindBestMatchingUnit(sample);
        Console.WriteLine($"{Format(sample)} -> grid unit ({winner % model.Width}, {winner / model.Width})");
    }
    Require(final < initial / 5, "SOM quantization error did not decrease sufficiently.");
}

static void AutoencoderDemo()
{
    var model = new Autoencoder(2, 1, 42);
    double[][] inputs = [[0, 1], [1, 0]];
    Train("Autoencoder reconstruction", model, inputs, inputs);
    foreach (var input in inputs) Console.WriteLine($"Encoding {Format(input)} -> {Format(model.Encode(input))}");
}
