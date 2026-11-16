using System.Globalization;
using NeuralNetworks;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
if (args.Length > 1 || (args.Length == 1 && args[0] is not ("train" or "infer")))
{
    Console.Error.WriteLine("Usage: dotnet run --project examples/Xor -- [train|infer]");
    return 2;
}
var network = new DenseNetwork([2, 4, 1], seed: 42);
double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
double[][] targets = [[0], [1], [1], [0]];
if (args.Length == 1 && args[0] == "infer")
{
    // Explicit illustrative parameters, not weights learned by the training mode.
    // Four hidden neurons: OR, NAND, and two unused units. Output approximates AND.
    network.SetParameters([
        20, 20, -10,
        -20, -20, 30,
        0, 0, 0,
        0, 0, 0,
        20, 20, 0, 0, -30
    ]);
    Console.WriteLine("Inference from fixed illustrative parameters (no training).");
}
else
{
    Console.WriteLine($"Initial mean loss: {MeanLoss():F6}");
    for (var epoch = 0; epoch < 10000; epoch++)
        for (var sample = 0; sample < inputs.Length; sample++)
            network.TrainSample(inputs[sample], targets[sample], learningRate: 0.5);
    Console.WriteLine($"Final mean loss:   {MeanLoss():F6}");
}
var success = true;
for (var i = 0; i < inputs.Length; i++)
{
    var prediction = network.Predict(inputs[i])[0];
    Console.WriteLine($"{inputs[i][0]} XOR {inputs[i][1]} -> {prediction:F6} (expected {targets[i][0]})");
    success &= Math.Abs(prediction - targets[i][0]) < 0.08;
}
return success ? 0 : 1;

double MeanLoss() => inputs.Select((input, i) => network.Loss(input, targets[i])).Average();
