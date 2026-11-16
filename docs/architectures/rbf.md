# Radial basis function network (RBF)

## How it works

An RBF network describes an input by its similarity to a set of centers. A Gaussian feature is close to 1 near its center and decays toward 0 with distance:

```text
φ_k(x) = exp(-½ Σ_i ((x_i - center_ki) / width)²)
output_j = sigmoid(b_j + Σ_k weight_jk φ_k(x))
```

The centers define local regions of input space; the output head combines them into predictions. In this implementation centers and a shared positive width are fixed. Only output weights and biases are trained by squared-error SGD. This is a sigmoid-output RBF variant, not the linear least-squares regression variant.

## How it was programmed

[RadialBasisNetwork.cs](../../src/NeuralNetworks/RadialBasisNetwork.cs) validates and copies every center, computes normalized squared distances, exponentiates the negative distances and applies a learned linear block followed by sigmoid. The [reverse-mode tape](../TRAINING.md) supplies gradients to the output head. `GetParameters()` contains the head only; fixed centers and width are construction arguments.

## How to use it

```csharp
using NeuralNetworks;

var model = new RadialBasisNetwork(
    centers: [[0, 0], [0, 1], [1, 0], [1, 1]],
    width: 0.4, outputSize: 1, seed: 42);
double[][] x = [[0, 0], [0, 1], [1, 0], [1, 1]];
double[][] target = [[0], [1], [1], [0]];
for (var epoch = 0; epoch < 1200; epoch++)
    for (var i = 0; i < x.Length; i++)
        model.TrainSample(x[i], target[i], 0.5);
var prediction = model.Predict([0, 1]);
```

Run `dotnet run --project examples/Architectures -- rbf`. It learns the four XOR cases using one center per corner and verifies loss reduction and prediction tolerance.

## Validation and limitations

Tests check a known Gaussian/head calculation, finite-difference head gradients, copied centers, invalid centers/widths and XOR learning. The example lowers mean loss from about `0.189836` to `0.000573`.

There is no k-means center selection, center/width optimization or least-squares solver. The caller must choose centers and scaling; too-small widths give negligible features away from centers, while too-large widths make features similar. Widths and centers must be finite, with a strictly positive width. Extreme normalized distances may be rejected as overflow before exponentiation.

Next: [SOM](som.md) · [all architecture guides](../README.md).
