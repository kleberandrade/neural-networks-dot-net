# Multilayer perceptron (MLP)

## How it works

An MLP is a dense feedforward network: every neuron receives all activations from the preceding layer. A hidden nonlinear layer lets the network represent functions such as XOR that a single linear boundary cannot separate.

```text
a⁰ = input
zˡ_j = bˡ_j + Σ_i wˡ_ji aˡ⁻¹_i
aˡ_j = sigmoid(zˡ_j)
loss = ½ Σ_j (aᴸ_j - target_j)²
```

Backpropagation computes output error signals, passes them backward through old weights and sigmoid derivatives, then forms weight and bias gradients. SGD subtracts a scaled gradient. See [the complete derivation](../MATHEMATICS.md).

## How it was programmed

[DenseNetwork.cs](../../src/NeuralNetworks/DenseNetwork.cs) implements forward propagation and analytic backpropagation explicitly, without the tape used by the other differentiable models. Activation arrays hold each layer's intermediate values. The topology is copied, weights use seeded Xavier initialization, and candidate updates replace parameters atomically after finite checks. Stable sigmoid evaluation avoids exponential overflow.

An array such as `[2, 4, 1]` means two inputs, four hidden neurons, one output. More hidden layers and multiple outputs work with the same code. `[2, 1]` is a sigmoid unit without a hidden layer, rather than the threshold-based `Perceptron` class.

## How to use it

```csharp
using NeuralNetworks;

var model = new DenseNetwork([2, 4, 1], seed: 42);
double[][] x = [[0, 0], [0, 1], [1, 0], [1, 1]];
double[][] target = [[0], [1], [1], [0]];
for (var epoch = 0; epoch < 10000; epoch++)
    for (var i = 0; i < x.Length; i++)
        model.TrainSample(x[i], target[i], learningRate: 0.5);
var xor = model.Predict([0, 1]);
```

Run `dotnet run --project examples/Architectures -- mlp`, or the more detailed original `dotnet run --project examples/Xor -- train`. For inference without training, use `dotnet run --project examples/Xor -- infer`; it uses explicitly hand-constructed parameters.

## Validation and limitations

Tests compare every gradient against central differences through multiple hidden layers and outputs, check known forward values, verify loss descent and learn all four XOR cases. The seeded example lowers mean loss from approximately `0.135352` to `0.000085`.

All activations are sigmoid, with bounded targets and squared error. Saturation can make gradients vanish. There are no selectable activation functions, softmax, batch optimizer or convergence guarantees. See [API details](../API.md) and [common contracts](../TRAINING.md).

Further reading: [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0).

Next: [autoencoder](autoencoder.md) · [all architecture guides](../README.md).
