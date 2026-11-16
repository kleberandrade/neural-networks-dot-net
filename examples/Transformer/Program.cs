using System.Globalization;
using NeuralNetworks;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
var mode = args.Length == 0 ? "all" : args[0];
if (args.Length > 1 || mode is not ("all" or "encoder" or "seq2seq"))
{
    Console.Error.WriteLine("Usage: dotnet run --project examples/Transformer -- [all|encoder|seq2seq]");
    return 2;
}
try
{
    if (mode is "all" or "encoder") Encoder();
    if (mode is "all" or "seq2seq") SequenceToSequence();
    return 0;
}
catch (Exception error) { Console.Error.WriteLine($"FAIL: {error.Message}"); return 1; }

static string Format(double[] values) => "[" + string.Join(", ", values.Select(x => x.ToString("F4", CultureInfo.InvariantCulture))) + "]";
static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

static void Encoder()
{
    Console.WriteLine("Multi-head encoder: two numerical intensity sequences.");
    var model = new TransformerEncoder(2, 1, 4, 1, seed: 42, headCount: 2);
    double[][] inputs = [[0, 0], [1, 1]];
    double[][] targets = [[0], [1]];
    double MeanLoss() => inputs.Select((x, i) => model.Loss(x, targets[i])).Average();
    var initial = MeanLoss();
    for (var epoch = 0; epoch < 1200; epoch++)
        for (var i = 0; i < inputs.Length; i++) model.TrainSample(inputs[i], targets[i], 0.5);
    var final = MeanLoss();
    Console.WriteLine($"Mean loss: {initial:F6} -> {final:F6}; heads={model.HeadCount}");
    Require(final < initial / 10, "Encoder loss did not decrease sufficiently.");
    for (var i = 0; i < inputs.Length; i++)
    {
        var output = model.Predict(inputs[i]);
        Console.WriteLine($"{Format(inputs[i])} -> {Format(output)}; target {Format(targets[i])}");
        Require(Math.Abs(output[0] - targets[i][0]) < 0.2, "Encoder prediction missed tolerance.");
    }
    var heads = model.GetMultiHeadAttentionWeights([0, 1]);
    for (var head = 0; head < heads.Length; head++)
        Console.WriteLine($"Head {head}, query 0 attention: {Format(heads[head][0])}");
}

static void SequenceToSequence()
{
    Console.WriteLine("Encoder-decoder: learn to reverse every two-bit numerical sequence.");
    var model = new TransformerSequenceToSequence(2, 2, 1, 1, modelSize: 4, headCount: 2, feedForwardSize: 8, seed: 42);
    double[][] source = [[0, 0], [0, 1], [1, 0], [1, 1]];
    double[][] target = [[0, 0], [1, 0], [0, 1], [1, 1]];
    // Shift right: BOS=0.5, then the previous true output. Never pass the current target as input.
    var decoderInput = target.Select(t => new[] { 0.5, t[0] }).ToArray();
    double MeanLoss() => source.Select((s, i) => model.Loss(s, decoderInput[i], target[i])).Average();
    var initial = MeanLoss();
    for (var epoch = 0; epoch < 1500; epoch++)
        for (var i = 0; i < source.Length; i++) model.TrainSample(source[i], decoderInput[i], target[i], 0.1);
    var final = MeanLoss();
    Console.WriteLine($"Teacher-forced mean sequence loss: {initial:F6} -> {final:F6}");
    Require(final < initial / 10, "Sequence loss did not decrease sufficiently.");
    for (var i = 0; i < source.Length; i++)
    {
        // Inference receives source and BOS only. Later decoder inputs are the model's own outputs.
        var generated = model.Generate(source[i], [0.5]);
        Console.WriteLine($"{Format(source[i])} -> {Format(generated)}; reverse target {Format(target[i])}");
        Require(generated.Zip(target[i]).All(x => Math.Abs(x.First - x.Second) < 0.2), "Autoregressive reversal missed tolerance.");
    }
    var attention = model.GetAttentionWeights(source[1], decoderInput[1]);
    Console.WriteLine($"Decoder head 0, query 0: {Format(attention.DecoderSelf[0][0])} (future key is masked)");
    Console.WriteLine($"Cross head 0, query 0:   {Format(attention.Cross[0][0])}");
    Console.WriteLine("Synthetic training cases only; no claim of unseen-sequence generalization or language generation.");
}
