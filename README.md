# Neural Networks in C# / .NET

[![C# 14](https://img.shields.io/badge/C%23-14-239120)](https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-14)
[![.NET 10 LTS](https://img.shields.io/badge/.NET-10%20LTS-512BD4)](https://dotnet.microsoft.com/en-us/download/dotnet/10.0)
[![MIT license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![CI](https://github.com/kleberandrade/neural-networks-dot-net/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/kleberandrade/neural-networks-dot-net/actions/workflows/ci.yml)

A small educational library that implements a dense feedforward neural network from scratch. Read the complete forward pass, backpropagation and stochastic gradient descent in [one class](src/NeuralNetworks/DenseNetwork.cs), then run XOR to see the network learn a nonlinear function.

This is a learning project, with explicit loops and `double` arrays. It is not a production machine-learning framework.

## Requirements

- A supported OS for the [.NET 10 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/10.0).
- .NET **10 LTS** SDK, with the latest servicing update. `global.json` selects an installed stable 10.0 SDK using `latestFeature` roll-forward.
- Git to clone the repository. No GPU, Python, external datasets or NuGet packages are required.

.NET 10 LTS is supported through November 14, 2028 according to [Microsoft's support policy](https://dotnet.microsoft.com/en-us/platform/support/policy). Builds restore only SDK reference packs; `NuGet.Config` intentionally clears package sources. Install the SDK before building offline.

## Install and quick start

```sh
git clone https://github.com/kleberandrade/neural-networks-dot-net.git
cd neural-networks-dot-net
dotnet restore
dotnet build --configuration Release --no-restore
dotnet run --project examples/Xor --configuration Release --no-build -- train
```

The seeded `2 → 4 → 1` network trains for 10,000 epochs, taking four SGD steps per epoch in a fixed order. A validated run with SDK 10.0.404 produced:

```text
Initial mean loss: 0.135352
Final mean loss:   0.000085
0 XOR 0 -> 0.009774 (expected 0)
0 XOR 1 -> 0.986943 (expected 1)
1 XOR 0 -> 0.986834 (expected 1)
1 XOR 1 -> 0.015576 (expected 0)
```

Small floating-point differences can occur across platforms or runtime versions. The example exits with code 1 if any output differs from its target by 0.08 or more. This check measures the four training cases, not generalization to an unseen dataset.

## Use the library

Reference the source project from your own .NET 10 application:

```sh
dotnet add path/to/YourApp.csproj reference src/NeuralNetworks/NeuralNetworks.csproj
```

```csharp
using NeuralNetworks;

var network = new DenseNetwork([2, 4, 1], seed: 42);
double[] input = [0, 1];
double[] target = [1];
var before = network.Loss(input, target);
var preUpdateLoss = network.TrainSample(input, target, learningRate: 0.5);
var prediction = network.Predict(input); // detached double[]
```

Call `TrainSample` repeatedly over your dataset. A single call performs one gradient step; it does not fit a dataset or guarantee convergence. All layers, including the output, use sigmoid activations. Inputs must be finite; targets must be finite and within `[0, 1]`.

## Examples

- **Learning XOR:** `dotnet run --project examples/Xor -- train` trains from seeded random weights and prints mean loss before and after training.
- **Inference only:** `dotnet run --project examples/Xor -- infer` loads explicitly constructed illustrative XOR parameters and predicts without training. These parameters are not a saved training result.
- **Inspect gradients:** `ComputeGradient(input, target)` returns an independent gradient vector without updating the model. The tests compare it to central finite differences.
- **Parameter snapshots:** `GetParameters()` and `SetParameters(values)` copy arrays. See [the API guide](docs/API.md) for ordering and validation.

## Tests and CI

```sh
dotnet build --configuration Release
dotnet run --project tests/NeuralNetworks.Tests --configuration Release --no-build
dotnet run --project examples/Xor --configuration Release --no-build -- train
dotnet run --project examples/Xor --configuration Release --no-build -- infer
dotnet format --verify-no-changes --no-restore
```

The test project is a package-free executable that prints each result and exits nonzero on failure. **`dotnet test` does not run this suite.** Tests cover known forward arithmetic, finite differences for every weight and bias, multiple hidden layers and outputs, loss reduction, XOR learning, deterministic seeding, array ownership, invalid dimensions/values, sigmoid saturation and atomic overflow rejection.

[CI](.github/workflows/ci.yml) runs these commands on Linux, macOS and Windows. Local validation is recorded in [the implementation log](docs/WORK_LOG.md). The installed ub-mode quality tools do not support C#, so no automated CRAP, duplication, coverage or mutation score is reported.

## Design and limitations

See [mathematics and glossary](docs/MATHEMATICS.md) and [API details](docs/API.md).

- Dense layers only; sigmoid and half squared-error loss only. No softmax, convolution, automatic differentiation engine, mini-batch optimizer or GPU support.
- Sequential per-sample SGD. Dataset iteration, normalization, validation splits, stopping criteria and shuffling belong to the caller. Learning rates and seeds affect convergence.
- Sigmoid can saturate and produce vanishing gradients. Use appropriately scaled inputs; this API does not normalize them.
- Finite values can still overflow during arithmetic. Such operations throw `ArithmeticException`; a rejected training step leaves the model unchanged.
- Each evaluation allocates arrays for clarity. Training mutates the network. An instance must not be trained, updated or read concurrently with mutation; use independent instances or external synchronization.
- Seeds reproduce initialization on the same runtime implementation, not a permanent cross-version serialization contract. There is no model file format or built-in persistence.
- Memory usage grows with the number of weights and activations. Impossible topology arithmetic throws `OverflowException`; excessively large allocations may throw `OutOfMemoryException`.
- No accuracy, throughput or suitability guarantees for real-world datasets.

## License

[MIT](LICENSE), copyright © 2018 Kléber de Oliveira Andrade. The original notice is preserved. The library and examples have no third-party runtime dependencies; .NET and GitHub Actions retain their own licenses.
